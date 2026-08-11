# detect_node 检测节点架构说明

> 本文档用大白话讲清楚 `src/detect/src/detect_node.cpp` 的整个架构：
> 每个函数干什么、输入什么、输出什么、对应代码在哪一行。
> 阅读顺序建议：先看第 1 节，然后看第 3 节（核心流水线），其他按需查阅。

---

## 1. 这个节点是干嘛的（30 秒版）

**输入**：摄像头图像（从话题 `/sensor_img` 或 `/sensor_img/compressed` 收）
**处理**：用 YOLOv8 模型（ONNX 格式）检测画面里的装甲板，再用 PnP 解算装甲板在机器人坐标系下的坐标
**输出**：三个话题——

| 话题 | 类型 | 内容 |
|---|---|---|
| `/detect/image` | sensor_msgs/Image | 画了绿色检测框 + 机器人坐标信息的图像（给人看的） |
| `/detect/detections` | detect/msg/DetectionArray | 结构化检测结果（给机器用的） |
| `/aim_target` | aim_interfaces/msg/AimInfo | 机器人坐标系下装甲板的坐标（mm）+ 图案类型（训练任务要求） |

```
image_publisher 节点 ──发图──▶  detect_node（本节点）──发画框图──▶ rviz/显示
                                       │
                                       ├──发检测结果──▶ 下游决策/瞄准节点
                                       └──发瞄准目标──▶ /aim_target（训练评测）
```

---

## 2. 数据流总览（一张图看懂整个文件）

```
main()                          程序入口，启动节点，进入事件循环
  │
  ▼ 收到一张图像
imageCallback / compressedImageCallback    回调：把 ROS 消息变成 cv::Mat
  │
  ▼
processFrame()                  ★ 核心流水线，一帧图从头走到尾
  │  ├─ ① letterbox()      任意尺寸图 → 640×640 带灰边的图
  │  ├─ ② infer()          640×640 图  → 42000 个数字（模型原始输出）
  │  ├─ ③ postprocess()    42000 个数字 → 几个干净检测框（原图坐标）
  │  ├─ ④ solveArmorPosition() + cameraToRobot()   检测框 → 机器人坐标系坐标
  │  ├─ ⑤ drawBoxes() + drawRobotPosition()  在副本上画绿框和坐标文字
  │  ├─ ⑥ 发布画框图        /detect/image
  │  ├─ ⑦ publishDetectionResults()   → /detect/detections
  │  ├─ ⑧ publishAimInfo()            → /aim_target
  │  └─ ⑨ saveResults()    （可选）把画框图 + CSV 汇总存到 save_dir
```

**全文件有 14 个函数**，其中真正要理解的是 `processFrame` 和它的子函数（①②③④），其余都是模板/辅助。

---

## 3. 函数逐个讲

### 3.0 main() — 程序入口（第 444 行）

```cpp
int main(int argc, char** argv)
```

| 项 | 内容 |
|---|---|
| 输入 | 命令行参数（argc/argv） |
| 输出 | 退出码 |
| 职责 | 初始化 ROS → 创建节点 → 进入事件循环 `rclcpp::spin`（阻塞，等图像到来触发回调） |

**大白话**：开机、插上电源、等着。回调什么时候触发、触发几次，全由 ROS 内部调度。

---

### 3.1 构造函数 DetectNode() — 初始化（第 82 行）

| 项 | 内容 |
|---|---|
| 输入 | 无 |
| 输出 | 无（把节点对象准备好） |
| 职责 | ① 读 ROS 参数（模型 + 相机标定）② 加载 ONNX 模型 ③ 预计算旋转矩阵和装甲板 3D 模型点 ④ 创建 2 个订阅者 + 3 个发布者 |

**基础 5 个参数**（启动时可覆写）：

| 参数名 | 默认值 | 含义 |
|---|---|---|
| `model_path` | .../best.onnx | 模型文件路径 |
| `conf_threshold` | 0.25 | 置信度阈值：低于它认为是误检 |
| `iou_threshold` | 0.45 | NMS 重叠阈值：重叠超过它算同一个目标 |
| `input_size` | 640 | 模型输入边长 |
| `verbose` | false | 是否每帧打印耗时 |

**相机标定参数**见 §3.13（内参、畸变、相机→机器人外参、图案类型）。

**大白话**：启动时把所有家当准备好——模型装进内存、把相机内外参换算成旋转矩阵和装甲板 3D 模型点、订阅/发布的话题接好线。模型文件加载失败会直接抛异常，节点起不来。

---

### 3.2 imageCallback() / compressedImageCallback() — 收图回调（第 126 / 138 行）

| 项 | imageCallback | compressedImageCallback |
|---|---|---|
| 输入 | Image 消息（原始像素） | CompressedImage 消息（JPEG 字节） |
| 输出 | 调 processFrame() | 调 processFrame() |
| 关键操作 | `cv_bridge::toCvShare`：免拷贝地把消息描述成 cv::Mat | `cv::imdecode`：解压 JPEG 得到 cv::Mat |

**大白话**：两个"门卫"，收到图先验一下货、转成 OpenCV 认识的格式，然后统一交给 processFrame。为什么订阅两个？因为 image_publisher 只发其中一个，接两个保证无论它发哪种都能收到。

---

### 3.3 processFrame() — 核心流水线（第 152 行）★最重要

| 项 | 内容 |
|---|---|
| 输入 | frame（cv::Mat，BGR 图）、stamp（时间戳）、frame_id（坐标系名） |
| 输出 | 发布两个话题 |
| 职责 | 把一帧图的完整处理流程串起来 |

**内部 7 步**（这就是整个节点的"主干"）：

```
1. letterbox          → 得到 640×640 输入图 + 变换参数(scale, pad_x, pad_y)
2. infer              → 得到 42000 个数字（原始输出）
3. postprocess        → 得到检测框列表（原图坐标）
4. drawBoxes          → 在副本上画框
5. 发布画框图          → /detect/image
6. 发布结构化结果      → /detect/detections（含原图时间戳）
7. verbose 时打印耗时
```

**大白话**：一条流水线，原料进（一帧图），成品出（两个话题消息）。每个步骤都是单独函数，好读好改。

---

### 3.4 letterbox() — 预处理（第 195 行）

| 项 | 内容 |
|---|---|
| 输入 | src（任意尺寸的 BGR 原图） |
| 输出（返回值） | 640×640 的 BGR 图（等比缩放 + 灰边） |
| 输出（出参） | scale（缩放比例）、pad_x / pad_y（灰边像素数） |
| 职责 | 把任意尺寸图"塞进"模型要求的 640×640，且不变形 |

**内部 5 步**：算缩放比例 → 算新尺寸 → 算灰边 → 建灰色画布 → 缩放图贴到画布中央。

```
原图 1280×960           处理完 640×640
┌──────────┐          ┌────────────────┐
│          │          │ 灰边 80px      │
│          │  → 0.5倍  ├────────────────┤
│          │          │ 640×480 画面    │
└──────────┘          ├────────────────┤
                      │ 灰边 80px      │
                      └────────────────┘
```

**为什么出参要带出去**：模型输出的框坐标是"640 空间"的，后处理要靠 `(坐标 - 填充) ÷ scale` 还原回原图坐标。**scale/pad 是后处理的钥匙，不能丢。**

**关键点**：灰边值 114 必须和模型训练时一致；取缩放比例用 min（保证不裁剪、不变形）。

---

### 3.5 infer() — 推理（第 226 行）

| 项 | 内容 |
|---|---|
| 输入 | input（640×640 的 BGR cv::Mat） |
| 输出 | output（std::vector\<float\>，42000 个数字） |
| 职责 | 把图像数据整理成模型要的格式，跑一次前向传播，把结果拷贝出来 |

**内部 5 步**：

| 步骤 | 代码 | 大白话 |
|---|---|---|
| ① 格式转换 | `blobFromImage` | 图像 BGR 字节 → float 数组（0~1，RGB 顺序，1×3×640×640） |
| ② 包装张量 | `CreateTensor` | 把 float 数组"包"成 ORT 认识的张量（借用内存，不拷贝） |
| ③ 取节点名 | `GetInputNameAllocated` | 问模型输入/输出节点叫什么名字 |
| ④ **跑模型** | `session_->Run(...)` | **真正的推理就这一句** |
| ⑤ 拷贝结果 | `memcpy` | 把结果从 ORT 张量复制到自己数组里 |

**输出布局（关键）**：`1 × 5 × 8400`
- 8400 = 80×80 + 40×40 + 20×20 三层网格的锚点数（每层每个格子一个"锚点"）
- 每个锚点 5 个数字：`cx, cy, w, h`（框中心+宽高，640 空间）+ 1 个类别得分（装甲板）

**大白话**：①②③⑤ 都是"包装活"（全世界写 ORT 的人都复制同样的模板），只有 ④ 是真正的"思考"。这部分代码**不需要理解，需要背**。

---

### 3.6 postprocess() — 后处理（第 280 行）★逻辑最复杂

| 项 | 内容 |
|---|---|
| 输入 | output（42000 个数字）+ scale/pad_x/pad_y（letterbox 出参）+ 原图宽高 |
| 输出 | std::vector\<Detection\>（最终检测框列表） |
| 职责 | 把模型的"原始猜测"变成"干净结论" |

**内部 4 步**：

| 步骤 | 做什么 | 类比 |
|---|---|---|
| ① 解码+过滤 | 遍历 8400 个锚点，得分 < 0.25 的扔掉 | 8400 人举手，只留喊得响的 |
| ② 排序 | 按置信度从高到低排 | 按嗓门排队 |
| ③ NMS 去重 | 重叠超过 45% 的框只留最高分那个 | 同一个人别报两次 |
| ④ 映射回原图 | `(坐标 - 填充) ÷ scale` 换算坐标并裁剪 | 地图比例尺换算 |

**Detection 结构**（最终产物）：

```cpp
struct Detection {
    int class_id;      // 类别索引（0 = armor）
    float confidence;  // 置信度（0~1）
    cv::Rect box;      // 原图坐标的框（x, y, w, h）
};
```

**大白话**：模型会"过度热情"——一个目标周围可能报几十个框，后处理负责**去重、过滤、换算**，最后每个目标只剩一个干净框。

---

### 3.7 drawBoxes() — 画框（第 385 行）

| 项 | 内容 |
|---|---|
| 输入 | frame（cv::Mat，会被原地修改）、detections（检测框列表） |
| 输出 | 无（直接改传入的图像） |
| 职责 | 每个检测画绿框 + 框上方写 "armor 87%" 标签（深色底白字） |

**大白话**：纯视觉辅助，画给人看。注意调用处是 `frame.clone()` 的副本上画的，不污染原图。

---

### 3.8 publishDetectionResults() — 发布结构化结果（第 415 行）

| 项 | 内容 |
|---|---|
| 输入 | detections（内部结构体列表）+ stamp + frame_id |
| 输出 | 发布到 `/detect/detections` |
| 职责 | 把内部 Detection 结构体"翻译"成 ROS 消息发出去 |

**翻译对照表**：

| 内部结构体 | ROS 消息 |
|---|---|
| `class_id` | `det_msg.class_id` |
| —（查表得名字） | `det_msg.class_name` = "armor" |
| `confidence` | `det_msg.confidence` |
| `box.x / box.y / box.width / box.height` | `det_msg.x / y / width / height` |

**大白话**：内部数据只能自己用，别的节点只认 ROS 消息，所以建一个 `DetectionArray` 容器，逐个填、逐个塞进去，最后 publish。下游节点拿到数组就能直接算目标中心去瞄准。

---

### 3.9 solveArmorPosition() — PnP 位姿解算（第 456 行）

| 项 | 内容 |
|---|---|
| 输入 | box（原图像素坐标的检测框） |
| 输出 | position_cam（装甲板中心在相机坐标系下的坐标，米） |
| 职责 | 用 PnP 由 2D 检测框反推装甲板在相机前的 3D 位置 |

**内部 2 步**：
1. 检测框四角作为装甲板四角的 2D 投影点（顺序与 3D 模型点一致：左上→右上→右下→左下）
2. `cv::solvePnP`：用 3D 模型点 + 2D 角点 + 内参 + 畸变，解出 tvec

**关键点**：
- 3D 模型点是构造函数里预计算的装甲板灯条角（见下），单位米
- 内参、畸变都来自构造函数的标定参数
- 结果 tvec 就是装甲板中心在相机坐标系的位置（x 右 y 下 z 前）

### 3.10 cameraToRobot() — 坐标变换（第 488 行）

| 项 | 内容 |
|---|---|
| 输入 | position_cam（相机坐标系，x 右 y 下 z 前） |
| 输出 | position_robot（机器人坐标系，x 前 y 左 z 上） |
| 职责 | 把相机坐标系坐标变换到机器人坐标系 |

**内部 2 步**：
1. 坐标轴对齐：相机坐标 (x右, y下, z前) → (z, -x, -y)，变成"x 前 y 左 z 上"的对齐坐标
2. 旋转 + 平移：`p_robot = R * 对齐坐标 + t`

**旋转矩阵 R**（构造函数里预计算）：`R = Rz(yaw) * Ry(pitch) * Rx(roll)`

### 3.11 drawRobotPosition() — 绘制坐标信息（第 512 行）

在检测框上方用蓝色底白字打印 `robot: (x, y, z) mm`，让人直观看到解算结果。

### 3.12 publishAimInfo() — 发布瞄准目标（第 537 行）

| 项 | 内容 |
|---|---|
| 输入 | detections（取置信度最高的一个）+ 解算结果 |
| 输出 | 发布到 `/aim_target`（aim_interfaces/msg/AimInfo） |
| 职责 | 把机器人坐标（米 → 毫米）和图案类型装进消息发出去 |

**消息字段**：
- `coordinate`：`[x, y, z]`（毫米，Int16）
- `type`：图案类型，默认 7（哨兵装甲板期望输出 7）

**坑点**：Int16 装不下小数，所以米要乘 1000 转成毫米再取整。

### 3.13 相机标定参数（构造函数里声明）

| 参数名 | 默认值 | 含义 |
|---|---|---|
| `camera_fx / fy / cx / cy` | 1462.37 / 1469.68 / 398.59 / 110.69 | 相机内参（培训说明给定） |
| `dist_k1 ~ k3` | 0.003518 ... 0.0 | 畸变系数（k1,k2,p1,p2,k3） |
| `cam_to_robot_x/y/z` | 0.08 / 0.0 / 0.05 | 相机在机器人系中的平移（米） |
| `cam_to_robot_roll/pitch/yaw` | 0 / 60 / 20 | 相机相对机器人的旋转（度） |
| `armor_type` | 7 | 装甲板图案类型 |
| `save_dir` | 空 | 非空时把画框图（frame_N.png）和汇总（results.csv）保存到该文件夹 |

**坐标系约定（重要，面试可能问）**：
- 相机坐标系（OpenCV/PnP 输出）：x 右，y 下，z 前
- 机器人坐标系：x 前，y 左，z 上
- 旋转顺序：`R = Rz(yaw) * Ry(pitch) * Rx(roll)`，先绕 x 再绕 y 最后绕 z
- 装甲板 3D 模型点：灯条中心间距 16cm → x=±0.08m，灯条长度 8cm → y=±0.04m，z=0
- 所有角度用弧度计算，角度制只在参数里出现

## 4. 消息类型速查

| 类型 | 定义位置 | 字段 |
|---|---|---|
| detect/msg/Detection | src/detect/msg/Detection.msg | class_id, class_name, confidence, x, y, width, height |
| detect/msg/DetectionArray | src/detect/msg/DetectionArray.msg | header + detection[] |
| aim_interfaces/msg/AimInfo | src/aim_interfaces/msg/AimInfo.msg | coordinate（int16[]，毫米）, type（int16） |

---

## 5. 学习建议（怎么用这份文档）

1. **第一遍**：只看第 1、2 节 + 3.3（processFrame），建立主干概念
2. **第二遍**：分别细看 letterbox / postprocess（这俩是核心算法）+ §3.9~3.13（姿态解算与坐标变换），其余是模板
3. **不需要背的**：infer 里的 ORT API 调用（blobFromImage / CreateTensor / CreateCpu 的参数）、构造函数细节——都是固定模板，用的时候查文档
4. **改代码的入口**：
   - 想调识别灵敏度 → 改 `conf_threshold` 参数
   - 想换模型 → 改 `model_path` 参数
   - 想改输出内容 → 改 `publishDetectionResults` 或 msg 文件
   - 想改坐标解算 → 改 §3.9~3.10（PnP 解算、坐标变换）或 §3.13 的相机标定参数
   - 想加预处理（如对比度增强）→ 在 processFrame 第 1 步之前加
