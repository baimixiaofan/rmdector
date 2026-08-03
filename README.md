# Image Publisher - ROS 2 图像发布节点

## 📖 项目简介

`image_publisher` 用于从指定文件夹读取图像文件并发布到ROS话题。

## 📦 安装与编译

### 1. 克隆仓库
```bash
cd ~/ros2_ws/src
git clone <repository_url>
```

### 2. 编译包
```bash
cd ~/ros2_ws
colcon build --packages-select image_publisher
source install/setup.bash
```

## 🚀 快速开始

### 基础使用
```bash
# 使用默认参数启动（需要修改图片文件夹路径）
ros2 launch image_publisher image_publisher.launch.py

# 指定图片文件夹
ros2 launch image_publisher image_publisher.launch.py image_folder:=/path/to/your/images
```


## 🎛️ 参数配置

### Launch文件参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `image_folder` | string | "" | 图片文件夹路径（空则使用包内示例） |
| `publish_rate` | double | 1.0 | 发布频率（Hz），建议范围：0.1-10.0 |
| `loop` | bool | false | 是否循环播放图片 |
| `preload_images` | bool | true | 是否预加载图像到内存 |
| `use_compression` | bool | false | 是否发布压缩图像 |
| `queue_size` | int | 50 | 发布队列大小 |
| `resize_width` | int | 0 | 缩放宽度（0为不缩放） |
| `resize_height` | int | 0 | 缩放高度（0为不缩放） |

### 节点参数详解

#### 📁 `image_folder` - 图片文件夹
- **默认**: 使用包内 `images/` 文件夹的示例图片
- **自定义**: 指定包含图片文件的文件夹路径
- **支持格式**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

#### ⏱️ `publish_rate` - 发布频率
- **范围**: 0.1 - 10.0 Hz
- **推荐**: 1.0 Hz（适合大多数应用）
- **高频**: 5.0+ Hz（实时应用）
- **低频**: 0.1-0.5 Hz（节省资源）

#### 🔄 `loop` - 循环播放
- `true`: 播放完所有图片后重新开始
- `false`: 播放完毕后停止发布

## 📊 话题信息

### 发布的话题

| 话题名 | 消息类型 | 说明 |
|--------|----------|------|
| `/image_raw` | `sensor_msgs/msg/Image` | 原始图像数据 |
| `/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | 压缩图像数据（可选） |

### 消息格式
```bash
# 查看话题信息
ros2 topic info /image_raw
ros2 topic echo /image_raw --no-arr

# 查看发布频率
ros2 topic hz /image_raw
```

## 🐛 故障排除

### 常见问题

#### 1. 找不到图片文件夹
```
错误: 无法加载图片文件，节点将退出
```
**解决方案:**
- 检查文件夹路径是否正确
- 确保文件夹包含支持的图像格式
- 检查文件读取权限

#### 2. 内存不足
```
错误: 预加载图像失败
```
**解决方案:**
- 设置 `preload_images:=false`
- 减少图片数量或尺寸
- 使用图像压缩

#### 3. 发布频率过高
```
警告: 发布队列已满
```
**解决方案:**
- 降低 `publish_rate` 参数
- 增加 `queue_size` 参数
- 启用图像压缩

#### 4. OpenCV错误
```
错误: cv_bridge异常
```
**解决方案:**
- 检查图像文件是否损坏
- 验证OpenCV安装
- 尝试不同的图像格式