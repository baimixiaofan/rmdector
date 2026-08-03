#include "detect/detect_node.hpp"

namespace detect
{

/**
 * @file detect_node.cpp
 * @brief YOLOv8 装甲板检测节点实现
 *
 * 模型输入输出说明（best.onnx）:
 *   - 输入: images  (1, 3, 640, 640) float32，RGB 归一化
 *   - 输出: output0 (1, 5, 8400) float32
 *     - 8400 = 640/8^2 + 640/16^2 + 640/32^2 个锚点
 *     - 前 4 行: cx, cy, w, h（640 空间像素坐标）
 *     - 后 1 行: 装甲板类别的概率（注意：此模型导出时已内置 sigmoid，
 *       输出直接是 0~1 概率，不要再套 sigmoid）
 *   - 类别: armor（只检测装甲板，不识别数字）
 */

/**
 * @brief 构造函数
 *
 * 声明 ROS 参数（模型路径、阈值、输入尺寸等），初始化 ONNX Runtime
 * 会话，创建图像订阅者与结果发布者
 */
DetectNode::DetectNode()
    : Node("detect_node")
{
    // ---- 参数 ----
    // declare_parameter: 声明 ROS 参数并给默认值，启动时可用
    // --ros-args -p xxx:=yyy 或 launch 文件覆写
    std::string model_path = this->declare_parameter(
        "model_path",
        std::string("/home/baimi/rmdector/src/detect/armor-4/weights/best.onnx"));
    conf_threshold_ = this->declare_parameter("conf_threshold", 0.25);  // 置信度过滤阈值
    iou_threshold_ = this->declare_parameter("iou_threshold", 0.45);    // NMS 去重阈值
    input_size_ = this->declare_parameter("input_size", 640);           // 模型输入边长
    verbose_ = this->declare_parameter("verbose", false);               // 是否打印每帧耗时

    // 类别名与训练 data.yaml 一致（单类：只检测装甲板，不识别数字）
    // 注意：必须和模型输出行数匹配（4 坐标 + 1 类别 = 5 行）
    class_names_ = {"armor"};

    // ---- 初始化 ONNX Runtime 会话 ----
    ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "detect");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);  // CPU 推理线程数（ONNX Runtime 内部并行）
    // 加载模型文件；失败会直接抛异常，节点起不来
    session_ = std::make_unique<Ort::Session>(*ort_env_, model_path.c_str(), session_options);

    // ---- 创建订阅者和发布者 ----
    // 同时订阅原始图和压缩图：image_publisher 只会发其中一个，另一个收不到数据不影响
    // 队列深度 1：图像只处理最新一帧，丢旧帧保实时性
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/sensor_img", 1,
        std::bind(&DetectNode::imageCallback, this, std::placeholders::_1));
    compressed_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        "/sensor_img/compressed", 1,
        std::bind(&DetectNode::compressedImageCallback, this, std::placeholders::_1));
    // /detect/image: 画好框的图像；/detect/detections: 结构化检测结果
    result_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/detect/image", 1);
    detection_pub_ = this->create_publisher<detect::msg::DetectionArray>("/detect/detections", 1);

    RCLCPP_INFO(this->get_logger(), "模型加载完成: %s", model_path.c_str());
    RCLCPP_INFO(this->get_logger(), "检测节点已启动, 订阅 /sensor_img, 发布 /detect/image 和 /detect/detections");
}

/**
 * @brief 原始图像话题订阅回调
 *
 * 通过 cv_bridge 将 ROS 图像消息转为 cv::Mat（bgr8）后交给 processFrame 处理
 * @param msg 收到的图像消息
 */
void DetectNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    // cv_bridge 把 ROS 图像消息转成 cv::Mat（默认 BGR8）
    try {
        cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;
        processFrame(frame, msg->header.stamp, msg->header.frame_id);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge异常: %s", e.what());
    }
}

/**
 * @brief 压缩图像话题订阅回调
 *
 * 直接用 cv::imdecode 将 JPEG 数据解压为 BGR，解码失败时仅告警跳过
 * @param msg 收到的压缩图像消息
 */
void DetectNode::compressedImageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
    // 压缩图直接用 cv::imdecode 解压成 BGR
    cv::Mat frame = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    if (frame.empty()) {
        RCLCPP_WARN(this->get_logger(), "压缩图解码失败");
        return;
    }
    processFrame(frame, msg->header.stamp, msg->header.frame_id);
}

/**
 * @brief 处理一帧图像：推理 + 后处理 + 画框 + 发布
 *
 * 处理流程：letterbox 预处理 → ONNX 推理 → 后处理（解码 + NMS）
 * → 在副本上画框并发布图像，同时发布结构化检测结果 DetectionArray
 * @param frame   待处理的 BGR 图像
 * @param stamp   消息时间戳
 * @param frame_id 消息坐标系/帧 id
 */
void DetectNode::processFrame(const cv::Mat& frame, const rclcpp::Time& stamp,
                              const std::string& frame_id)
{
    // 记录处理耗时：从收到帧到发布结果的总时间
    auto t0 = std::chrono::steady_clock::now();

    // 1. 预处理（letterbox）：等比缩放 + 灰边填充到 640x640
    //    scale/pad 记录变换参数，供第 3 步把检测框映射回原图坐标
    float scale;
    int pad_x, pad_y;
    cv::Mat input = letterbox(frame, scale, pad_x, pad_y);

    // 2. 推理：ONNX Runtime 前向传播，得到原始输出张量
    std::vector<float> output;
    infer(input, output);

    // 3. 后处理（解码 + NMS），得到原图像素坐标的检测框
    auto detections = postprocess(output, scale, pad_x, pad_y, frame.cols, frame.rows);

    // 4. 画框：在副本上画（不污染原图数据）
    cv::Mat annotated = frame.clone();
    drawBoxes(annotated, detections);

    // 5. 发布画框图像：cv::Mat 转成 sensor_msgs/Image 消息（bgr8 编码）
    cv_bridge::CvImage cv_image(std_msgs::msg::Header(), "bgr8", annotated);
    result_pub_->publish(*cv_image.toImageMsg());

    // 6. 发布结构化检测结果：把内存里的 Detection 逐个转成 ROS 消息
    //    供下游决策使用（如瞄准、跟踪），避免下游再解析图像
    detect::msg::DetectionArray array_msg;
    array_msg.header.stamp = stamp;
    array_msg.header.frame_id = frame_id;
    for (const auto& d : detections) {
        detect::msg::Detection det_msg;
        det_msg.class_id = d.class_id;                    // 类别索引
        det_msg.class_name = class_names_[d.class_id];    // 类别名字符串
        det_msg.confidence = d.confidence;                // 置信度
        det_msg.x = d.box.x;                              // 框左上角 x
        det_msg.y = d.box.y;                              // 框左上角 y
        det_msg.width = d.box.width;                      // 框宽
        det_msg.height = d.box.height;                    // 框高
        array_msg.detections.push_back(det_msg);
    }
    detection_pub_->publish(array_msg);

    // 7. 日志：打印目标数量和耗时，方便调试
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    RCLCPP_INFO(this->get_logger(), "检测到 %zu 个目标, 耗时 %ld ms",
                detections.size(), ms);
}

/**
 * @brief 预处理：等比缩放到模型输入尺寸并灰边填充（letterbox）
 *
 * 缩放比例取两个方向中的较小值，保证长边恰好等于 input_size_，
 * 短边方向居中放置，剩余区域用灰色 (114) 填充，保持图像比例不变
 * @param src   输入原图
 * @param scale 输出缩放比例（供后处理坐标还原使用）
 * @param pad_x 输出水平填充像素数（供后处理坐标还原使用）
 * @param pad_y 输出垂直填充像素数（供后处理坐标还原使用）
 * @return 尺寸为 input_size_ x input_size_ 的输入张量图像
 */
cv::Mat DetectNode::letterbox(const cv::Mat& src, float& scale, int& pad_x, int& pad_y)
{
    // 等比缩放：取两个方向缩放比例的较小值，保证长边恰好 = input_size_
    //（比如 1280x768 图，scale = 640/1280 = 0.5，高变为 384）
    scale = std::min(static_cast<float>(input_size_) / src.cols,
                     static_cast<float>(input_size_) / src.rows);
    int new_w = static_cast<int>(std::round(src.cols * scale));
    int new_h = static_cast<int>(std::round(src.rows * scale));
    // 填充量：短边方向两侧各留 (input_size_ - 新尺寸) / 2 的灰边
    pad_x = (input_size_ - new_w) / 2;
    pad_y = (input_size_ - new_h) / 2;

    // 建 640x640 灰色画布（RGB 都是 114，与 YOLO 训练时的填充值一致），
    // 缩放后的图放到画布中央，其余区域保持灰色
    cv::Mat resized, canvas(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(src, resized, cv::Size(new_w, new_h));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));
    return canvas;
}

/**
 * @brief ONNX 推理：输入 640x640 BGR 图，输出原始张量
 *
 * 内部完成 HWC BGR → NCHW float32 转换（swapRB 实现 BGR→RGB 并归一化
 * 到 0~1），创建输入张量运行会话，并把输出拷贝到 output
 * @param input  预处理后的模型输入图像
 * @param output 输出的原始推理结果（8400 x 类别数+4）
 */
void DetectNode::infer(const cv::Mat& input, std::vector<float>& output)
{
    // 图像格式转换：HWC BGR 8UC3 → NCHW float32
    // blobFromImage 参数: 缩放系数 1/255（归一化）、无缩放尺寸、无均值、
    // swapRB=true（BGR→RGB，模型是按 RGB 训练的）
    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar(), true);

    // 创建输入张量：把 blob 的数据指针包成 ONNX 张量
    // 注意只是"借用"blob 的内存，blob 必须活到 Run 结束（作用域内没问题）
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, 3, input_size_, input_size_};
    Ort::Value input_value = Ort::Value::CreateTensor<float>(
        mem_info, reinterpret_cast<float*>(blob.data), blob.total(), shape.data(), shape.size());

    // 获取模型输入/输出节点的名字（onnx 里的 "images" 和 "output0"）
    // GetInputNameAllocated 返回智能指针：名字字符串由 ORT 分配，
    // 必须先保存到局部变量再取 .get()，否则悬垂指针
    auto input_name = session_->GetInputNameAllocated(0, allocator_);
    auto output_name = session_->GetOutputNameAllocated(0, allocator_);
    std::array<const char*, 1> input_names{input_name.get()};
    std::array<const char*, 1> output_names{output_name.get()};
    // 执行前向传播：输入 1 个张量，输出 1 个张量
    auto results = session_->Run(Ort::RunOptions{nullptr},
                                 input_names.data(), &input_value, 1,
                                 output_names.data(), 1);

    // 把结果从 ORT 张量拷贝到 std::vector（类型转换 + 连续内存）
    size_t num_elements = results[0].GetTensorTypeAndShapeInfo().GetElementCount();
    output.resize(num_elements);
    std::memcpy(output.data(), results[0].GetTensorData<float>(), num_elements * sizeof(float));
}

/**
 * @brief 后处理：解码候选框 + 阈值过滤 + NMS，坐标映射回原图
 *
 * 输出布局 (1, 5, 8400)：行 0~3 是 cx,cy,w,h，行 4 是装甲板类别得分。
 * 三步流程：
 *   1. 解码所有候选框，取每个锚点得分最高的类别，低于 conf_threshold_ 的过滤；
 *   2. 按置信度降序排列，便于 NMS 依次贪心抑制；
 *   3. 类内 NMS（IoU 阈值 iou_threshold_），并把框从 640 空间映射回原图并裁剪。
 * @param output 模型原始输出张量
 * @param scale  letterbox 的缩放比例
 * @param pad_x  letterbox 的水平填充像素
 * @param pad_y  letterbox 的垂直填充像素
 * @param orig_w 原图宽度
 * @param orig_h 原图高度
 * @return 过滤后的检测结果列表
 */
std::vector<Detection> DetectNode::postprocess(const std::vector<float>& output,
                                               float scale, int pad_x, int pad_y,
                                               int orig_w, int orig_h)
{
    // 输出布局 (1, 5, 8400)：行 0~3 是 cx,cy,w,h，行 4 是装甲板类别得分。
    // 8400 个锚点按 3 个尺度排布：大网格(80x80=6400, 小目标)
    // + 中网格(40x40=1600) + 小网格(20x20=400, 大目标)。
    // 每个锚点存 5 个数字：位置(4) + 类别得分(1)
    const int num_classes = static_cast<int>(class_names_.size());
    const int num_anchors = static_cast<int>(output.size() / (4 + num_classes));

    // 第一步：解码所有候选框，过滤低置信度
    // 内存布局是"按行存"的：同一行(如第4行得分)的 8400 个数连续排列，
    // 所以取第 i 个锚点的第 c 类得分要用 output[(4+c)*num_anchors + i]
    struct Candidate
    {
        float cx, cy, w, h, score;
        int class_id;
    };
    std::vector<Candidate> candidates;
    for (int i = 0; i < num_anchors; ++i) {
        // 取该锚点在所有类别里的最高得分及对应类别
        float max_score = 0.0f;
        int best_class = 0;
        for (int c = 0; c < num_classes; ++c) {
            float s = output[(4 + c) * num_anchors + i];
            if (s > max_score) {
                max_score = s;
                best_class = c;
            }
        }
        // 此模型的输出已内置 sigmoid，得分直接就是置信度（0~1）
        // 启发判断：得分落在 [0,1] 视为概率直接用；否则当作 logits 再套 sigmoid
        // （防御性代码：防止换了没内置 sigmoid 的模型后结果全错）
        float confidence = (max_score >= 0.0f && max_score <= 1.0f) ? max_score
                            : 1.0f / (1.0f + std::exp(-max_score));
        // 置信度低于阈值直接丢弃（调低阈值可提高召回，但误检增多）
        if (confidence < conf_threshold_)
            continue;

        // 保存候选：框中心 + 宽高（640 空间坐标）+ 得分 + 类别
        candidates.push_back({
            output[0 * num_anchors + i],
            output[1 * num_anchors + i],
            output[2 * num_anchors + i],
            output[3 * num_anchors + i],
            confidence,
            best_class
        });
    }

    // 第二步：按置信度降序排列，便于 NMS 依次贪心抑制
    // （高置信度的框先被保留，低置信度且重叠的框被它压掉）
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    // 第三步：类内 NMS，并把框从 640 空间映射回原图
    // NMS = 同一目标周围会有多个重叠框，只保留得分最高的那个
    std::vector<Detection> results;
    std::vector<bool> suppressed(candidates.size(), false);  // 标记被抑制的框
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i])
            continue;
        const auto& c = candidates[i];
        // 坐标还原：640 空间 → 原图
        // (640 坐标 - 填充) / 缩放比例 = 原图坐标（letterbox 的逆变换）
        float x1 = (c.cx - c.w / 2 - pad_x) / scale;
        float y1 = (c.cy - c.h / 2 - pad_y) / scale;
        float x2 = (c.cx + c.w / 2 - pad_x) / scale;
        float y2 = (c.cy + c.h / 2 - pad_y) / scale;
        // 裁剪到图像范围内，防止框超出画面
        x1 = std::max(0.0f, x1); y1 = std::max(0.0f, y1);
        x2 = std::min(static_cast<float>(orig_w), x2);
        y2 = std::min(static_cast<float>(orig_h), y2);

        results.push_back({c.class_id, c.score,
                           cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                    static_cast<int>(x2 - x1), static_cast<int>(y2 - y1))});

        // 把与当前框同类别、且 IoU 超过阈值的后续框全部抑制
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j] || candidates[j].class_id != c.class_id)
                continue;
            const auto& d = candidates[j];
            // IoU = 两框交集面积 / 两框并集面积，衡量重叠程度
            // 两个框在 640 空间下的 IoU（相对缩放，映射前后 IoU 不变）
            float ix1 = std::max(c.cx - c.w / 2, d.cx - d.w / 2);
            float iy1 = std::max(c.cy - c.h / 2, d.cy - d.h / 2);
            float ix2 = std::min(c.cx + c.w / 2, d.cx + d.w / 2);
            float iy2 = std::min(c.cy + c.h / 2, d.cy + d.h / 2);
            float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
            float union_area = c.w * c.h + d.w * d.h - inter;
            if (inter / union_area > iou_threshold_)
                suppressed[j] = true;
        }
    }
    return results;
}

/**
 * @brief 在图像上画检测框和标签
 *
 * 每个检测画绿色矩形框，框上方绘制 "类别 置信度%" 的深色标签
 * @param frame      待绘制的图像（原地修改）
 * @param detections 检测结果列表
 */
void DetectNode::drawBoxes(cv::Mat& frame, const std::vector<Detection>& detections)
{
    for (const auto& d : detections) {
        cv::Scalar color(0, 255, 0);  // 绿色框
        cv::rectangle(frame, d.box, color, 2);  // 画矩形框，线宽 2px

        // 标签: "armor 87%"（类别名 + 置信度百分比）
        std::string label = class_names_[d.class_id] + " " +
                            std::to_string(static_cast<int>(d.confidence * 100)) + "%";
        int baseline;
        // 测量文字大小，决定背景色块的尺寸（避免文字被框遮挡）
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        // 文字放在框的上方（离顶部太近时下移，避免超出画面）
        cv::Point text_origin(d.box.x, std::max(d.box.y - 5, text_size.height));
        // 先画深色背景块，再写白字，保证文字在任何背景下都清晰
        cv::rectangle(frame, cv::Rect(text_origin, text_size + cv::Size(4, baseline)),
                      color, cv::FILLED);
        cv::putText(frame, label, text_origin + cv::Point(2, text_size.height - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

} // namespace detect

/**
 * @brief 程序入口
 *
 * 初始化 ROS → 创建检测节点 → 进入事件循环（订阅回调在此触发）
 * @param argc 命令行参数个数
 * @param argv 命令行参数
 * @return 退出码
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<detect::DetectNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
