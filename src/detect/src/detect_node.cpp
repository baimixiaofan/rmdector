#include "detect/detect_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>

namespace detect
{

// ======================================================================
// 匿名命名空间：以下辅助工具仅本文件可见，不暴露到类接口
// ======================================================================
namespace
{

/**
 * @brief 展开路径开头的 ~（只支持 ~ 和 ~/xxx 两种形式）
 *
 * std::filesystem 不会展开 shell 风格的 ~：launch 或命令行直接传
 * "~/xxx" 时，会在当前目录下创建名为 "~" 的文件夹。这里手动用
 * HOME 环境变量替换，让路径落回用户主目录。
 */
std::string expandTilde(const std::string& path)
{
    // 不以 ~ 开头，或空路径，原样返回
    if (path.empty() || path[0] != '~') {
        return path;
    }
    const char* home = std::getenv("HOME");
    if (home == nullptr) {
        // 拿不到 HOME 时原样返回，避免破坏传入路径
        return path;
    }
    // "~" 单独出现时就是主目录本身，否则替换掉开头的 "~"
    if (path.size() == 1) {
        return std::string(home);
    }
    return std::string(home) + path.substr(1);
}

/** @brief 角度制转弧度 */
double toRadians(double deg)
{
    return deg * CV_PI / 180.0;
}

/** @brief 绕 X 轴的旋转矩阵（输入为弧度） */
cv::Mat rotationX(double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return (cv::Mat_<double>(3, 3) << 1, 0, 0,
                                     0, c, -s,
                                     0, s, c);
}

/** @brief 绕 Y 轴的旋转矩阵（输入为弧度） */
cv::Mat rotationY(double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return (cv::Mat_<double>(3, 3) << c, 0, s,
                                     0, 1, 0,
                                     -s, 0, c);
}

/** @brief 绕 Z 轴的旋转矩阵（输入为弧度） */
cv::Mat rotationZ(double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return (cv::Mat_<double>(3, 3) << c, -s, 0,
                                     s, c, 0,
                                     0, 0, 1);
}

/**
 * @brief 模型输出中的一个候选框（640 空间坐标）
 */
struct Candidate
{
    float cx;        // 框中心 x 坐标（letterbox 后，640 空间）
    float cy;        // 框中心 y 坐标（letterbox 后，640 空间）
    float w;         // 框宽（640 空间）
    float h;         // 框高（640 空间）
    float score;     // 置信度得分，范围 [0, 1]
    int class_id;    // 类别索引
};

/**
 * @brief 把模型的原始得分转换为置信度
 *
 * 本模型导出时已内置 sigmoid，得分直接就是 [0,1] 的概率。
 * 防御性处理：若得分不在 [0,1] 区间，说明换了不含 sigmoid 的模型，
 * 此时把得分当作 logits 再套一次 sigmoid。
 */
float toConfidence(float raw_score)
{
    const bool looks_like_probability = (raw_score >= 0.0f && raw_score <= 1.0f);
    if (looks_like_probability) {
        return raw_score;
    }
    return 1.0f / (1.0f + std::exp(-raw_score));
}

/**
 * @brief 计算两个候选框的 IoU（交并比）
 *
 * 在 640 空间下计算即可：等比缩放前后 IoU 不变。
 */
float intersectionOverUnion(const Candidate& a, const Candidate& b)
{
    // 两个框的边界坐标（640 空间）
    const float a_x1 = a.cx - a.w / 2.0f;
    const float a_y1 = a.cy - a.h / 2.0f;
    const float a_x2 = a.cx + a.w / 2.0f;
    const float a_y2 = a.cy + a.h / 2.0f;

    const float b_x1 = b.cx - b.w / 2.0f;
    const float b_y1 = b.cy - b.h / 2.0f;
    const float b_x2 = b.cx + b.w / 2.0f;
    const float b_y2 = b.cy + b.h / 2.0f;

    // 交集区域：两个框重叠的部分
    const float inter_x1 = std::max(a_x1, b_x1);
    const float inter_y1 = std::max(a_y1, b_y1);
    const float inter_x2 = std::min(a_x2, b_x2);
    const float inter_y2 = std::min(a_y2, b_y2);

    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;

    // 并集 = 两框面积之和 - 交集
    const float union_area = a.w * a.h + b.w * b.h - inter_area;
    return inter_area / union_area;
}

/**
 * @brief 比较两个候选框的置信度（sort 排序用）
 *
 * 返回 true 表示 a 应该排在 b 前面。
 * 规则：得分高的排前面（降序）。
 */
bool scoreHigher(const Candidate& a, const Candidate& b)
{
    return a.score > b.score;
}

} // namespace

// ======================================================================
// 构造函数
// ======================================================================
DetectNode::DetectNode()
    : Node("detect_node")
{
    // ---- 1. 读取 ROS 参数 ----
    // 启动时可用 --ros-args -p 参数名:=值 或 launch 文件覆写默认值
    const std::string model_path = expandTilde(this->declare_parameter(
        "model_path",
        std::string("/home/baimi/rmdector/src/detect/armor-4/weights/best.onnx")));
    conf_threshold_ = this->declare_parameter("conf_threshold", 0.25);  // 置信度过滤阈值
    iou_threshold_ = this->declare_parameter("iou_threshold", 0.45);    // NMS 去重阈值
    input_size_ = this->declare_parameter("input_size", 640);           // 模型输入边长
    verbose_ = this->declare_parameter("verbose", false);               // 是否打印每帧耗时
    // 结果保存文件夹（空=不保存），先把 ~ 展开成用户主目录，避免在 cwd 下建出名为 ~ 的文件夹
    save_dir_ = expandTilde(this->declare_parameter("save_dir", std::string("")));

    // 类别名需与训练 data.yaml 一致，行数必须匹配模型输出（4 坐标 + 1 类别）
    class_names_ = {"armor"};

    // ---- 1.5 相机标定与坐标变换参数（默认值取自 26 赛季培训说明）----
    armor_type_ = static_cast<int16_t>(this->declare_parameter("armor_type", 7));
    // 相机内参矩阵：fx, 0, cx; 0, fy, cy; 0, 0, 1
    const double fx = this->declare_parameter("camera_fx", 1462.3697);
    const double fy = this->declare_parameter("camera_fy", 1469.68385);
    const double cx = this->declare_parameter("camera_cx", 398.59394);
    const double cy = this->declare_parameter("camera_cy", 110.68997);
    // 相机畸变系数：k1, k2, p1, p2, k3
    const double k1 = this->declare_parameter("dist_k1", 0.003518);
    const double k2 = this->declare_parameter("dist_k2", -0.311778);
    const double p1 = this->declare_parameter("dist_p1", -0.016581);
    const double p2 = this->declare_parameter("dist_p2", 0.023682);
    const double k3 = this->declare_parameter("dist_k3", 0.0);
    // 相机在机器人坐标系下的平移（米）
    const double cam_tx = this->declare_parameter("cam_to_robot_x", 0.08);
    const double cam_ty = this->declare_parameter("cam_to_robot_y", 0.0);
    const double cam_tz = this->declare_parameter("cam_to_robot_z", 0.05);
    // 相机相对机器人的旋转（角度制）
    const double cam_roll = this->declare_parameter("cam_to_robot_roll", 0.0);
    const double cam_pitch = this->declare_parameter("cam_to_robot_pitch", 60.0);
    const double cam_yaw = this->declare_parameter("cam_to_robot_yaw", 20.0);

    camera_matrix_ = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                                                0, fy, cy,
                                                0, 0, 1);
    dist_coeffs_ = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
    cam_translation_ = cv::Point3f(static_cast<float>(cam_tx),
                                   static_cast<float>(cam_ty),
                                   static_cast<float>(cam_tz));
    // 相机系→机器人系旋转矩阵：先 roll 绕 x，再 pitch 绕 y，最后 yaw 绕 z
    cam_rotation_ = rotationZ(toRadians(cam_yaw)) *
                    rotationY(toRadians(cam_pitch)) *
                    rotationX(toRadians(cam_roll));

    // 装甲板 3D 模型点（单位：米），坐标原点取装甲板中心：
    //   两侧灯条中心间距 16cm -> x = ±0.08
    //   灯条长度 8cm          -> y = ±0.04
    //   z = 0（装甲板近似为平面）
    // 顺序：左上、右上、右下、左下（与检测框四角顺序一一对应）
    armor_points_ = {
        cv::Point3f(-0.08f, 0.04f, 0.0f),   // 左上
        cv::Point3f(0.08f, 0.04f, 0.0f),    // 右上
        cv::Point3f(0.08f, -0.04f, 0.0f),   // 右下
        cv::Point3f(-0.08f, -0.04f, 0.0f),  // 左下
    };

    // ---- 2. 初始化 ONNX Runtime 并加载模型 ----
    ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "detect");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);  // CPU 推理线程数
    session_ = std::make_unique<Ort::Session>(*ort_env_, model_path.c_str(), session_options);
    RCLCPP_INFO(this->get_logger(), "模型加载完成: %s", model_path.c_str());

    // ---- 3. 创建订阅者与发布者 ----
    // 原始图和压缩图都订阅：image_publisher 只会发其中一个，另一个收不到不影响。
    // 队列深度 1：只处理最新一帧，丢旧帧保实时性。
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/sensor_img", 1,
        std::bind(&DetectNode::imageCallback, this, std::placeholders::_1));
    compressed_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        "/sensor_img/compressed", 1,
        std::bind(&DetectNode::compressedImageCallback, this, std::placeholders::_1));

    result_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/detect/image", 1);
    detection_pub_ = this->create_publisher<detect::msg::DetectionArray>("/detect/detections", 1);
    aim_pub_ = this->create_publisher<aim_interfaces::msg::AimInfo>("/aim_target", 1);

    // ---- 4. 结果保存：建目录 + 打开汇总 CSV（save_dir 为空则跳过）----
    if (!save_dir_.empty()) {
        std::filesystem::create_directories(save_dir_);
        results_file_.open(save_dir_ + "/results.csv", std::ios::out | std::ios::trunc);
        if (results_file_.is_open()) {
            // CSV 表头：帧号, 检测数, 类别, 置信度, 框x, 框y, 框w, 框h, 机器人x(mm), 机器人y(mm), 机器人z(mm)
            results_file_ << "frame,count,class,confidence,x,y,w,h,robot_x_mm,robot_y_mm,robot_z_mm\n";
        }
        RCLCPP_INFO(this->get_logger(), "检测结果将保存到: %s", save_dir_.c_str());
    }

    RCLCPP_INFO(this->get_logger(),
                "检测节点已启动, 订阅 /sensor_img, 发布 /detect/image、/detect/detections 和 /aim_target");
}

// ======================================================================
// 图像订阅回调
// ======================================================================
void DetectNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    try {
        // toCvShare：共享像素内存（不拷贝），并把宽高步长描述成 cv::Mat
        cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;
        processFrame(frame, msg->header.stamp, msg->header.frame_id);
    } catch (cv_bridge::Exception& e) {
        // 消息编码无法转成 bgr8 时兜底：记录日志、跳过本帧，不让节点崩溃
        RCLCPP_ERROR(this->get_logger(), "cv_bridge异常: %s", e.what());
    }
}

void DetectNode::compressedImageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
    // 压缩图是 JPEG 字节流，需要先解码成像素（BGR）
    cv::Mat frame = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    if (frame.empty()) {
        RCLCPP_WARN(this->get_logger(), "压缩图解码失败");
        return;
    }
    processFrame(frame, msg->header.stamp, msg->header.frame_id);
}

// ======================================================================
// 单帧处理流水线：预处理 → 推理 → 后处理 → 画框 → 发布
// ======================================================================
void DetectNode::processFrame(const cv::Mat& frame, const rclcpp::Time& stamp,
                              const std::string& frame_id)
{
    const auto start_time = std::chrono::steady_clock::now();

    // ---- 1. 预处理：等比缩放 + 灰边填充到 input_size_ x input_size_ ----
    // 记录 scale / pad 参数，供第 3 步把检测框映射回原图坐标
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    const cv::Mat input = letterbox(frame, scale, pad_x, pad_y);

    // ---- 2. 推理：ONNX 前向传播，得到原始输出张量 ----
    std::vector<float> raw_output;
    infer(input, raw_output);

    // ---- 3. 后处理：解码 + 过滤 + NMS，得到原图像素坐标的检测框 ----
    const std::vector<Detection> detections =
        postprocess(raw_output, scale, pad_x, pad_y, frame.cols, frame.rows);

    // ---- 4. 姿态解算：对置信度最高的目标解算机器人坐标系坐标 ----
    // 本任务只关心最优目标（其余检测框仅用于显示）
    cv::Point3f position_cam;
    cv::Point3f position_robot;
    const bool solved = (!detections.empty() &&
                         solveArmorPosition(detections[0].box, position_cam));
    if (solved) {
        cameraToRobot(position_cam, position_robot);
    }

    // ---- 5. 在副本上画框（不污染原图数据），并打印坐标信息 ----
    cv::Mat annotated = frame.clone();
    drawBoxes(annotated, detections);
    if (solved) {
        drawRobotPosition(annotated, position_robot, detections[0].box);
    }

    // ---- 6. 发布画框图像（沿用原图时间戳与坐标系，方便下游同步）----
    std_msgs::msg::Header image_header;
    image_header.stamp = stamp;
    image_header.frame_id = frame_id;
    cv_bridge::CvImage cv_image(image_header, "bgr8", annotated);
    result_pub_->publish(*cv_image.toImageMsg());

    // ---- 7. 发布结构化检测结果（供下游瞄准 / 决策使用） ----
    publishDetectionResults(detections, stamp, frame_id);

    // ---- 8. 发布瞄准目标信息（机器人坐标系坐标 + 图案类型） ----
    publishAimInfo(detections, solved, position_robot);

    // ---- 8.5 保存检测结果到文件夹（供离线检查，save_dir 为空则跳过）----
    if (!save_dir_.empty()) {
        saveResults(annotated, detections, solved, position_robot);
    }

    // ---- 9. 按参数决定是否打印每帧耗时 ----
    if (verbose_) {
        const long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();
        RCLCPP_INFO(this->get_logger(), "检测到 %zu 个目标, 耗时 %ld ms",
                    detections.size(), elapsed_ms);
    }
}

// ======================================================================
// 预处理：letterbox（等比缩放 + 灰边填充）
// ======================================================================
cv::Mat DetectNode::letterbox(const cv::Mat& src, float& scale, int& pad_x, int& pad_y)
{
    // 1. 计算等比缩放比例：取宽、高两个方向中较小的值，
    //    保证长边恰好等于 input_size_，短边一定不超出
    //    （例如 1280x960 图：scale = 640/1280 = 0.5，高缩放后为 480）
    scale = std::min(static_cast<float>(input_size_) / src.cols,
                     static_cast<float>(input_size_) / src.rows);

    // 2. 计算缩放后的新尺寸
    const int new_w = static_cast<int>(std::round(src.cols * scale));
    const int new_h = static_cast<int>(std::round(src.rows * scale));

    // 3. 计算灰边填充量：短边方向两侧各补 (input_size_ - 新尺寸) / 2
    pad_x = (input_size_ - new_w) / 2;
    pad_y = (input_size_ - new_h) / 2;

    // 4. 创建 input_size_ x input_size_ 的灰色画布
    //    填充值 114 与 YOLO 训练时的填充色一致（RGB 三个通道都是 114）
    cv::Mat canvas(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));

    // 5. 等比缩放原图，并粘贴到画布中央
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

    return canvas;
}

// ======================================================================
// 推理：把预处理后的图像送入模型，输出原始张量
// ======================================================================
void DetectNode::infer(const cv::Mat& input, std::vector<float>& output)
{
    // ---------- 1. 把图像改成模型认识的样子 ----------
    // cv::dnn::blobFromImage：一步把"宽x高x3通道的 BGR 图像"变成"1张x3通道x640x640 的 float 数组"
    // 参数从右往左看：
    //   true          → 交换 B 和 R 通道（BGR 变 RGB），因为模型是按 RGB 训练的
    //   cv::Scalar()  → 不减去均值
    //   cv::Size()    → 不做额外缩放（letterbox 已经缩放好了）
    //   1.0/255.0     → 每个像素值除以 255，把 0~255 归一化成 0~1
    // 返回的 blob 是一个 4 维 float 数组（NCHW），但底层内存是连续的一长串数
    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar(), true);

    // ---------- 2. 把 blob 包装成 ONNX Runtime 认识的样子 ----------
    // 告诉 ORT：数据存在普通 CPU 内存里（不是显卡显存）
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);//cpu推理直接用这两个参数即可，除非你需要手动管理
    // 声明张量的形状：1 张图、3 个通道、宽 input_size_、高 input_size_（即 1, 3, 640, 640）
    const std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
    // CreateTensor：把 blob 7那段连续内存"包"成一个 ONNX 张量对象
    // 注意：没有复制数据，只是把指针递进去借用；blob 必须活到 Run 执行完
    Ort::Value input_value = Ort::Value::CreateTensor<float>(
        mem_info, reinterpret_cast<float*>(blob.data), blob.total(),
         input_shape.data(), input_shape.size());
// Ort::Value::CreateTensor<float>(
//     mem_info,                              // ① 内存说明书（CPU 内存）
//     reinterpret_cast<float*>(blob.data),   // ② 数据首地址指针
//     blob.total(),                          // ③ 元素总数（1228800 个 float）
//     input_shape.data(),                    // ④ 形状数组 {1,3,640,640}
//     input_shape.size());                   // ⑤ 维度数（4）
// 5 个参数：放哪 → 在哪 → 多少 → 形状 → 几维。

    // ---------- 3. 问模型要输入/输出的名字 ----------
    // 模型文件里给输入节点起的名字是 "images"，输出节点是 "output0"，
    // 但代码不硬编码，直接问 ORT 要（0 号输入/输出就是唯一的那个）
    auto input_name = session_->GetInputNameAllocated(0, allocator_);
    // 同上，要输出节点名字；返回的是智能指针，名字内存由 ORT 管理
    auto output_name = session_->GetOutputNameAllocated(0, allocator_);
    // Run 需要的参数是 const char* 数组，这里把智能指针里的名字指针取出来装进数组
    // 之所以要先保存 input_name/output_name 再取 .get()，是因为智能指针销毁名字就没了
    const std::array<const char*, 1> input_names{input_name.get()};
    const std::array<const char*, 1> output_names{output_name.get()};

    // ---------- 4. 真正跑模型 ----------
    // Run 就是"前向传播"：把输入张量喂进模型，算出输出张量
    // 参数依次是：运行选项（用默认）、输入名字数组、输入张量指针、输入个数、输出名字数组、输出个数
    auto results = session_->Run(Ort::RunOptions{nullptr},
                                 input_names.data(), &input_value, 1,
                                 output_names.data(), 1);

    // ---------- 5. 把结果从 ORT 张量拷到 std::vector ----------
    // 问结果张量一共有多少个元素（1 * 5 * 8400 = 42000 个 float）
    const size_t num_elements = results[0].GetTensorTypeAndShapeInfo().GetElementCount();
    // 把 output 数组扩到能装下 42000 个 float
    output.resize(num_elements);
    // 内存拷贝：从 ORT 张量的数据指针，原样复制 42000*4 字节到 output 里
    // 之后就完全脱离 ORT，可以像普通数组一样访问了
    std::memcpy(output.data(), results[0].GetTensorData<float>(), num_elements * sizeof(float));
}

// ======================================================================
// 后处理：解码候选框 → 置信度过滤 → NMS 去重 → 映射回原图
// ======================================================================
std::vector<Detection> DetectNode::postprocess(const std::vector<float>& output,
                                               float scale, int pad_x, int pad_y,
                                               int orig_w, int orig_h)
{
    // 模型输出布局 (1, 5, 8400)：
    //   行 0~3: cx, cy, w, h（640 空间的像素坐标）
    //   行 4  : 类别得分
    //   8400 = 80x80 + 40x40 + 20x20 三个尺度的锚点数量（大网格测小目标）
    const int num_classes = static_cast<int>(class_names_.size());  //类别名称的个数
    const int num_anchors = static_cast<int>(output.size() / (4 + num_classes));  //计算锚点数量

    // ---- 第一步：解码所有候选框，过滤低置信度 ----
    // 输出按"行"连续存储：第 i 个锚点的第 c 类得分位于 output[(4 + c) * num_anchors + i]
    std::vector<Candidate> candidates;
    candidates.reserve(num_anchors);
    for (int anchor = 0; anchor < num_anchors; ++anchor) {
        // 取该锚点在所有类别中得分最高的类别
        float max_score = 0.0f;
        int best_class = 0;
        for (int c = 0; c < num_classes; ++c) {
            const float score = output[(4 + c) * num_anchors + anchor];
            if (score > max_score) {
                max_score = score;
                best_class = c;
            }
        }

        // 得分转置信度，低于阈值直接丢弃（调低阈值可提高召回，但误检增多）
        const float confidence = toConfidence(max_score);
        if (confidence < conf_threshold_) {
            continue;
        }

        candidates.push_back({
            output[0 * num_anchors + anchor],
            output[1 * num_anchors + anchor],
            output[2 * num_anchors + anchor],
            output[3 * num_anchors + anchor],
            confidence,
            best_class,
        });
    }//这里就是在找出一个锚点置信度最高的类，然后这个最高的类如果置信度也低就直接丢掉


    // ---- 第二步：按置信度降序排列 ----
    // 高置信度框先被保留，与它重叠的低置信度框被它压掉
    // 排序规则见 scoreHigher：得分高的排前面
    std::sort(candidates.begin(), candidates.end(), scoreHigher);

    // ---- 第三步：类内 NMS，确定要保留的候选 ----
    // 同一目标周围会产生多个重叠框，NMS 只保留得分最高的那个
    std::vector<Candidate> kept;
    std::vector<bool> suppressed(candidates.size(), false);  // 标记被抑制的框
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        const Candidate& current = candidates[i];
        kept.push_back(current);

        // 抑制掉与当前框同类别、且 IoU 超过阈值的后续框
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            if (candidates[j].class_id != current.class_id) {
                continue;  // 不同类别不抑制，避免误杀
            }
            if (intersectionOverUnion(current, candidates[j]) > iou_threshold_) {
                suppressed[j] = true;
            }
        }
    }

    // ---- 第四步：把保留的框从 640 空间映射回原图，并裁剪到画面内 ----
    // letterbox 逆变换：(640 坐标 - 填充) / 缩放比例 = 原图坐标
    std::vector<Detection> results;
    results.reserve(kept.size());
    for (const Candidate& c : kept) {
        const float x1 = (c.cx - c.w / 2.0f - pad_x) / scale;
        const float y1 = (c.cy - c.h / 2.0f - pad_y) / scale;
        const float x2 = (c.cx + c.w / 2.0f - pad_x) / scale;
        const float y2 = (c.cy + c.h / 2.0f - pad_y) / scale;

        // 裁剪到图像范围内，防止框超出画面
        const float clipped_x1 = std::max(0.0f, x1);
        const float clipped_y1 = std::max(0.0f, y1);
        const float clipped_x2 = std::min(static_cast<float>(orig_w), x2);
        const float clipped_y2 = std::min(static_cast<float>(orig_h), y2);

        results.push_back({
            c.class_id,
            c.score,
            cv::Rect(static_cast<int>(clipped_x1), static_cast<int>(clipped_y1),
                     static_cast<int>(clipped_x2 - clipped_x1),
                     static_cast<int>(clipped_y2 - clipped_y1)),
        });
    }

    return results;
}

// ======================================================================
// 画框：绿色矩形 + 深色底白字标签
// ======================================================================
void DetectNode::drawBoxes(cv::Mat& frame, const std::vector<Detection>& detections)
{
    const cv::Scalar box_color(0, 255, 0);  // 绿色

    for (const Detection& det : detections) {
        cv::rectangle(frame, det.box, box_color, 2);  // 线宽 2px

        // 标签文本: "armor 87%"（类别名 + 置信度百分比）
        const std::string label = class_names_[det.class_id] + " " +
                                  std::to_string(static_cast<int>(det.confidence * 100)) + "%";

        // 测量文字尺寸，决定背景色块的大小
        int baseline = 0;
        const cv::Size text_size =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // 文字放在框的上方（离顶部太近时下移，避免超出画面）
        const cv::Point text_origin(det.box.x, std::max(det.box.y - 5, text_size.height));

        // 先画深色背景块，再写白字，保证任何背景下都清晰
        cv::rectangle(frame, cv::Rect(text_origin, text_size + cv::Size(4, baseline)),
                      box_color, cv::FILLED);
        cv::putText(frame, label, text_origin + cv::Point(2, text_size.height - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// ======================================================================
// 发布结构化检测结果（把内存结构转成 ROS 消息）
// ======================================================================
void DetectNode::publishDetectionResults(const std::vector<Detection>& detections,
                                         const rclcpp::Time& stamp,
                                         const std::string& frame_id)
{
    detect::msg::DetectionArray array_msg;
    array_msg.header.stamp = stamp;        // 与原图同一时刻
    array_msg.header.frame_id = frame_id;  // 原图坐标系

    // 逐个把内部结构体翻译成 ROS 消息
    for (const Detection& det : detections) {
        detect::msg::Detection det_msg;
        det_msg.class_id = det.class_id;                    // 类别索引
        det_msg.class_name = class_names_[det.class_id];    // 类别名字符串
        det_msg.confidence = det.confidence;                // 置信度
        det_msg.x = det.box.x;                              // 框左上角 x
        det_msg.y = det.box.y;                              // 框左上角 y
        det_msg.width = det.box.width;                      // 框宽
        det_msg.height = det.box.height;                    // 框高
        array_msg.detections.push_back(det_msg);
    }

    detection_pub_->publish(array_msg);
}

// ======================================================================
// 姿态解算：PnP 求装甲板中心在相机坐标系下的坐标
// ======================================================================
bool DetectNode::solveArmorPosition(const cv::Rect& box, cv::Point3f& position_cam) const
{
    // ---- 1. 检测框四角作为装甲板四角的 2D 投影点 ----
    // YOLO 检测框紧贴装甲板（含两侧灯条），四角顺序与 3D 模型点一致
    const std::vector<cv::Point2f> image_points = {
        cv::Point2f(box.x, box.y),                             // 左上
        cv::Point2f(box.x + box.width, box.y),                 // 右上
        cv::Point2f(box.x + box.width, box.y + box.height),    // 右下
        cv::Point2f(box.x, box.y + box.height),                // 左下
    };

    // ---- 2. PnP 解算：得到装甲板中心在相机坐标系下的坐标（米）----
    // 相机坐标系（OpenCV）：tvec = (x 右, y 下, z 前)
    cv::Mat rvec;
    cv::Mat tvec;
    const bool ok = cv::solvePnP(armor_points_, image_points,
                                 camera_matrix_, dist_coeffs_,
                                 rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    if (!ok) {
        return false;
    }

    position_cam = cv::Point3f(static_cast<float>(tvec.at<double>(0)),
                               static_cast<float>(tvec.at<double>(1)),
                               static_cast<float>(tvec.at<double>(2)));
    return true;
}

// ======================================================================
// 坐标变换：相机坐标系 -> 机器人坐标系
// ======================================================================
void DetectNode::cameraToRobot(const cv::Point3f& position_cam, cv::Point3f& position_robot) const
{
    // 相机系（x 右 y 下 z 前）先转成与机器人系对齐的坐标（x 前 y 左 z 上）：
    //   相机 x(右) -> 机器人 -y，相机 y(下) -> 机器人 -z，相机 z(前) -> 机器人 x
    const cv::Mat aligned = (cv::Mat_<double>(3, 1) << position_cam.z,
                                                     -position_cam.x,
                                                     -position_cam.y);

    // 旋转 + 平移 -> 机器人坐标系
    const cv::Mat t_robot = (cv::Mat_<double>(3, 1) << cam_translation_.x,
                                                       cam_translation_.y,
                                                       cam_translation_.z);
    const cv::Mat robot = cam_rotation_ * aligned + t_robot;

    position_robot = cv::Point3f(static_cast<float>(robot.at<double>(0)),
                                 static_cast<float>(robot.at<double>(1)),
                                 static_cast<float>(robot.at<double>(2)));
}

// ======================================================================
// 绘制机器人坐标信息
// ======================================================================
void DetectNode::drawRobotPosition(cv::Mat& frame, const cv::Point3f& position_robot,
                                   const cv::Rect& box)
{
    // 米 -> 毫米，显示为整数便于阅读
    const int x_mm = static_cast<int>(std::round(position_robot.x * 1000.0));
    const int y_mm = static_cast<int>(std::round(position_robot.y * 1000.0));
    const int z_mm = static_cast<int>(std::round(position_robot.z * 1000.0));
    const std::string label = "robot: (" + std::to_string(x_mm) + ", " +
                              std::to_string(y_mm) + ", " +
                              std::to_string(z_mm) + ") mm";

    // 放在检测框上方（若与置信度标签重叠则下移一行）
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int label_bottom = std::max(box.y - 5, text_size.height);
    const cv::Point text_origin(box.x, std::max(label_bottom - text_size.height - 4, text_size.height));

    cv::rectangle(frame, cv::Rect(text_origin, text_size + cv::Size(4, baseline)),
                  cv::Scalar(255, 0, 0), cv::FILLED);
    cv::putText(frame, label, text_origin + cv::Point(2, text_size.height - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

// ======================================================================
// 发布瞄准目标信息：机器人坐标系坐标 + 图案类型 -> /aim_target
// ======================================================================
void DetectNode::publishAimInfo(const std::vector<Detection>& detections,
                                bool solved, const cv::Point3f& position_robot)
{
    aim_interfaces::msg::AimInfo aim_msg;
    aim_msg.type = armor_type_;  // 哨兵装甲板图案期望输出 7

    // 只在成功解算时填充坐标，否则保持空数组（下游可据此判断无目标）
    if (solved && !detections.empty()) {
        // 米 -> 毫米（Int16 数组无法表示小数），四舍五入取整
        aim_msg.coordinate = {
            static_cast<int16_t>(std::round(position_robot.x * 1000.0)),
            static_cast<int16_t>(std::round(position_robot.y * 1000.0)),
            static_cast<int16_t>(std::round(position_robot.z * 1000.0)),
        };
    }

    aim_pub_->publish(aim_msg);
}

// ======================================================================
// 保存检测结果：画框图像 + 汇总 CSV
// ======================================================================
void DetectNode::saveResults(const cv::Mat& annotated,
                             const std::vector<Detection>& detections,
                             bool solved, const cv::Point3f& position_robot)
{
    // 帧号从 1 开始递增，作为文件名编号，保证保存顺序与播放顺序一致
    ++frame_counter_;
    const std::string prefix = save_dir_ + "/frame_" +
                               std::to_string(frame_counter_);

    // ---- 1. 保存画框图像 ----
    cv::imwrite(prefix + ".png", annotated);

    // ---- 2. 追加一行汇总记录到 CSV ----
    if (!results_file_.is_open()) {
        return;
    }
    results_file_ << frame_counter_ << "," << detections.size() << ","
                  << class_names_[detections.empty() ? 0 : detections[0].class_id] << ","
                  << (detections.empty() ? 0.0f : detections[0].confidence) << ","
                  << (detections.empty() ? 0 : detections[0].box.x) << ","
                  << (detections.empty() ? 0 : detections[0].box.y) << ","
                  << (detections.empty() ? 0 : detections[0].box.width) << ","
                  << (detections.empty() ? 0 : detections[0].box.height) << ","
                  << (solved ? static_cast<int>(std::round(position_robot.x * 1000.0)) : 0) << ","
                  << (solved ? static_cast<int>(std::round(position_robot.y * 1000.0)) : 0) << ","
                  << (solved ? static_cast<int>(std::round(position_robot.z * 1000.0)) : 0) << "\n";
    results_file_.flush();
}

} // namespace detect

// ======================================================================
// 程序入口
// ======================================================================
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try {
        // 模型文件不存在或损坏时 Ort::Session 会抛异常，
        // 这里兜底打印原因后正常退出，而不是留下一个晦涩的崩溃栈
        auto node = std::make_shared<detect::DetectNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("detect_node"), "节点启动失败: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
