#ifndef DETECT_NODE_HPP
#define DETECT_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "detect/msg/detection.hpp"
#include "detect/msg/detection_array.hpp"
#include "aim_interfaces/msg/aim_info.hpp"

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
 *   /aim_target              机器人坐标系下的装甲板坐标 (aim_interfaces/AimInfo)
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

    /**
     * @brief 把检测结果转换成 ROS 消息并发布到 /detect/detections
     * @param detections 检测结果列表
     * @param stamp   消息时间戳
     * @param frame_id 消息坐标系/帧 id
     */
    void publishDetectionResults(const std::vector<Detection>& detections,
                                 const rclcpp::Time& stamp, const std::string& frame_id);

    /**
     * @brief 用 PnP 解算检测框对应的装甲板中心在相机坐标系下的坐标
     * @param box           原图像素坐标系下的装甲板检测框
     * @param position_cam  输出：装甲板中心在相机坐标系下的坐标（米，x 右 y 下 z 前）
     * @return 解算成功返回 true
     */
    bool solveArmorPosition(const cv::Rect& box, cv::Point3f& position_cam) const;

    /**
     * @brief 把相机坐标系下的坐标变换到机器人坐标系
     * @param position_cam   相机坐标系坐标（米，x 右 y 下 z 前）
     * @param position_robot 输出：机器人坐标系坐标（米，x 前 y 左 z 上）
     */
    void cameraToRobot(const cv::Point3f& position_cam, cv::Point3f& position_robot) const;

    /**
     * @brief 在图像上绘制机器人坐标系下的目标坐标文本
     * @param frame          待绘制的图像（原地修改）
     * @param position_robot 机器人坐标系下的坐标（米）
     * @param box            检测框（文本显示在框上方）
     */
    void drawRobotPosition(cv::Mat& frame, const cv::Point3f& position_robot, const cv::Rect& box);

    /**
     * @brief 发布 AimInfo 到 /aim_target（机器人坐标系下的装甲板坐标 + 图案类型）
     * @param detections     检测结果列表（取置信度最高的一个）
     * @param solved         本帧是否成功解算出坐标
     * @param position_robot 解算出的机器人坐标系坐标（米）
     */
    void publishAimInfo(const std::vector<Detection>& detections,
                        bool solved, const cv::Point3f& position_robot);

    /**
     * @brief 把画框图像和检测结果保存到 save_dir_ 文件夹（供离线检查识别效果）
     * @param annotated      画框后的图像
     * @param detections     检测结果列表
     * @param solved         本帧是否成功解算出坐标
     * @param position_robot 解算出的机器人坐标系坐标（米）
     */
    void saveResults(const cv::Mat& annotated, const std::vector<Detection>& detections,
                     bool solved, const cv::Point3f& position_robot);

    // ROS 接口
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;             ///< 原始图像订阅者
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_sub_; ///< 压缩图像订阅者
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr result_pub_;               ///< 检测结果图像发布者
    rclcpp::Publisher<detect::msg::DetectionArray>::SharedPtr detection_pub_;        ///< 结构化检测结果发布者
    rclcpp::Publisher<aim_interfaces::msg::AimInfo>::SharedPtr aim_pub_;             ///< 瞄准目标信息发布者

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

    // 检测结果保存
    std::string save_dir_;            ///< 结果保存文件夹（空 = 不保存）
    unsigned long frame_counter_ = 0; ///< 已处理帧计数（用于文件名编号）
    std::ofstream results_file_;      ///< 检测结果汇总 CSV 文件流

    // 相机标定与坐标变换参数（默认值取自 26 赛季培训说明）
    int16_t armor_type_;              ///< 装甲板图案类型（哨兵期望输出 7）
    cv::Mat camera_matrix_;           ///< 相机内参矩阵 3x3
    cv::Mat dist_coeffs_;             ///< 相机畸变系数 (k1,k2,p1,p2,k3)
    cv::Mat cam_rotation_;            ///< 相机系→机器人系旋转矩阵 3x3
    cv::Point3f cam_translation_;     ///< 相机系→机器人系平移（米）
    std::vector<cv::Point3f> armor_points_;  ///< 装甲板 4 个灯条角的 3D 模型点（米）
};

} // namespace detect

#endif // DETECT_NODE_HPP
