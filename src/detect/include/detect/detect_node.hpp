#ifndef DETECT_NODE_HPP
#define DETECT_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "detect/msg/detection.hpp"
#include "detect/msg/detection_array.hpp"

namespace detect
{

/**
 * @brief 单个目标检测结果（内存结构）
 *
 * 由后处理函数产出，用于画框显示，不直接用于 ROS 通信
 * （ROS 侧使用自定义消息 detect/msg/Detection）
 */
struct Detection
{
    int class_id;        ///< 类别索引，对应 class_names_ 中的名称
    float confidence;    ///< 置信度得分，范围 [0, 1]
    cv::Rect box;        ///< 在原图坐标系下的检测框
};

/**
 * @brief YOLOv8 装甲板检测节点
 *
 * 订阅图像话题 → ONNX Runtime 推理 → 画框并发布结果
 *
 * 输入话题:
 *   /sensor_img              (sensor_msgs/Image, BGR8)
 *   /sensor_img/compressed   (sensor_msgs/CompressedImage, JPEG)
 *
 * 输出话题:
 *   /detect/image            画了检测框的图像
 *   /detect/detections       结构化检测结果 (DetectionArray)
 */
class DetectNode : public rclcpp::Node
{
public:
    /**
     * @brief 构造函数
     * 声明参数、创建订阅/发布者、加载 ONNX 模型并初始化会话
     */
    DetectNode();

    /**
     * @brief 析构函数
     */
    ~DetectNode() = default;

private:
    /**
     * @brief 原始图像话题订阅回调
     * @param msg 收到的图像消息
     */
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

    /**
     * @brief 压缩图像话题订阅回调
     * @param msg 收到的压缩图像消息
     */
    void compressedImageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);

    /**
     * @brief 处理一帧图像：推理 + 后处理 + 画框 + 发布
     * @param frame   待处理的 BGR 图像
     * @param stamp   消息时间戳
     * @param frame_id 消息坐标系/帧 id
     */
    void processFrame(const cv::Mat& frame, const rclcpp::Time& stamp, const std::string& frame_id);

    /**
     * @brief 预处理：等比缩放到模型输入尺寸并灰边填充（letterbox）
     * @param src   输入原图
     * @param scale 输出缩放比例（供后处理坐标还原使用）
     * @param pad_x 输出水平填充像素数（供后处理坐标还原使用）
     * @param pad_y 输出垂直填充像素数（供后处理坐标还原使用）
     * @return 尺寸为 input_size_ x input_size_ 的输入张量图像
     */
    cv::Mat letterbox(const cv::Mat& src, float& scale, int& pad_x, int& pad_y);

    /**
     * @brief ONNX 推理：输入 640x640 BGR 图，输出原始张量
     * @param input  预处理后的模型输入图像
     * @param output 输出的原始推理结果（8400 x 类别数+4）
     */
    void infer(const cv::Mat& input, std::vector<float>& output);

    /**
     * @brief 后处理：解码候选框 + 阈值过滤 + NMS，坐标映射回原图
     * @param output 模型原始输出张量
     * @param scale  letterbox 的缩放比例
     * @param pad_x  letterbox 的水平填充像素
     * @param pad_y  letterbox 的垂直填充像素
     * @param orig_w 原图宽度
     * @param orig_h 原图高度
     * @return 过滤后的检测结果列表
     */
    std::vector<Detection> postprocess(const std::vector<float>& output,
                                       float scale, int pad_x, int pad_y,
                                       int orig_w, int orig_h);

    /**
     * @brief 在图像上画检测框和标签
     * @param frame      待绘制的图像（原地修改）
     * @param detections 检测结果列表
     */
    void drawBoxes(cv::Mat& frame, const std::vector<Detection>& detections);

    // ROS 接口
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;             ///< 原始图像订阅者
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_sub_; ///< 压缩图像订阅者
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr result_pub_;               ///< 检测结果图像发布者
    rclcpp::Publisher<detect::msg::DetectionArray>::SharedPtr detection_pub_;        ///< 结构化检测结果发布者

    // ONNX Runtime 会话
    std::unique_ptr<Ort::Env> ort_env_;            ///< ONNX Runtime 环境
    std::unique_ptr<Ort::Session> session_;        ///< 推理会话
    Ort::AllocatorWithDefaultOptions allocator_;   ///< 默认分配器
    std::vector<std::string> class_names_;         ///< 类别名称列表（对应模型输出索引）

    // 参数
    int input_size_;          ///< 模型输入边长（640）
    double conf_threshold_;   ///< 置信度阈值
    double iou_threshold_;    ///< NMS IoU 阈值
    bool verbose_;            ///< 是否打印每帧耗时
};

} // namespace detect

#endif // DETECT_NODE_HPP
