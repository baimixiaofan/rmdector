#include "detect/detect_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace detect
{

// ======================================================================
// 匿名命名空间：以下辅助工具仅本文件可见，不暴露到类接口
// ======================================================================
namespace
{

/**
 * @brief 模型输出中的一个候选框（640 空间坐标）
 */
struct Candidate
{
    float cx, cy;   // 框中心坐标（letterbox 后，640 空间）
    float w, h;     // 框宽高（640 空间）
    float score;    // 置信度得分，范围 [0, 1]
    int class_id;   // 类别索引
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

} // namespace

// ======================================================================
// 构造函数
// ======================================================================
DetectNode::DetectNode()
    : Node("detect_node")
{
    // ---- 1. 读取 ROS 参数 ----
    // 启动时可用 --ros-args -p 参数名:=值 或 launch 文件覆写默认值
    const std::string model_path = this->declare_parameter(
        "model_path",
        std::string("/home/baimi/rmdector/src/detect/armor-4/weights/best.onnx"));
    conf_threshold_ = this->declare_parameter("conf_threshold", 0.25);  // 置信度过滤阈值
    iou_threshold_ = this->declare_parameter("iou_threshold", 0.45);    // NMS 去重阈值
    input_size_ = this->declare_parameter("input_size", 640);           // 模型输入边长
    verbose_ = this->declare_parameter("verbose", false);               // 是否打印每帧耗时

    // 类别名需与训练 data.yaml 一致，行数必须匹配模型输出（4 坐标 + 1 类别）
    class_names_ = {"armor"};

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

    RCLCPP_INFO(this->get_logger(),
                "检测节点已启动, 订阅 /sensor_img, 发布 /detect/image 和 /detect/detections");
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

    // ---- 4. 在副本上画框（不污染原图数据） ----
    cv::Mat annotated = frame.clone();
    drawBoxes(annotated, detections);

    // ---- 5. 发布画框图像 ----
    cv_bridge::CvImage cv_image(std_msgs::msg::Header(), "bgr8", annotated);
    result_pub_->publish(*cv_image.toImageMsg());

    // ---- 6. 发布结构化检测结果（供下游瞄准 / 决策使用） ----
    publishDetectionResults(detections, stamp, frame_id);

    // ---- 7. 按参数决定是否打印每帧耗时 ----
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
    // 1. 格式转换：HWC BGR 8UC3 → NCHW float32
    //    blobFromImage 参数：归一化 1/255、不缩放、无均值、swapRB=true（BGR→RGB，
    //    模型按 RGB 训练）
    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar(), true);

    // 2. 把 blob 包装成 ONNX 输入张量
    //    注意只是借用 blob 的内存，blob 必须存活到 Run 结束（本函数作用域内没问题）
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
    Ort::Value input_value = Ort::Value::CreateTensor<float>(
        mem_info, reinterpret_cast<float*>(blob.data), blob.total(),
        input_shape.data(), input_shape.size());

    // 3. 获取模型输入 / 输出节点的名字（onnx 里的 "images" 和 "output0"）
    //    GetInputNameAllocated 返回智能指针：名字由 ORT 分配，
    //    必须先保存到局部变量再取 .get()，否则悬垂指针
    auto input_name = session_->GetInputNameAllocated(0, allocator_);
    auto output_name = session_->GetOutputNameAllocated(0, allocator_);
    const std::array<const char*, 1> input_names{input_name.get()};
    const std::array<const char*, 1> output_names{output_name.get()};

    // 4. 前向传播：输入 1 个张量，输出 1 个张量
    auto results = session_->Run(Ort::RunOptions{nullptr},
                                 input_names.data(), &input_value, 1,
                                 output_names.data(), 1);

    // 5. 把结果从 ORT 张量拷贝到 std::vector（连续内存，方便后续访问）
    const size_t num_elements = results[0].GetTensorTypeAndShapeInfo().GetElementCount();
    output.resize(num_elements);
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
    const int num_classes = static_cast<int>(class_names_.size());
    const int num_anchors = static_cast<int>(output.size() / (4 + num_classes));

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
    }

    // ---- 第二步：按置信度降序排列 ----
    // 高置信度框先被保留，与它重叠的低置信度框被它压掉
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

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

} // namespace detect

// ======================================================================
// 程序入口
// ======================================================================
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<detect::DetectNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
