#ifndef IMAGE_PUBLISHER_NODE_HPP
#define IMAGE_PUBLISHER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/header.hpp>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <chrono>
#include <memory>

namespace image_publisher
{

/**
 * @brief 图像发布节点类
 * 
 * 该类负责从指定文件夹读取图像文件并发布到ROS话题
 * 支持多种图像格式、循环播放、图像压缩等功能
 */
class ImagePublisherNode : public rclcpp::Node
{
public:
    /**
     * @brief 构造函数
     * 初始化节点参数、发布者和定时器
     */
    ImagePublisherNode();

    /**
     * @brief 析构函数
     */
    ~ImagePublisherNode() = default;

private:
    /**
     * @brief 加载图片文件列表
     * @return 成功返回true，失败返回false
     */
    bool loadImageFiles();
    
    /**
     * @brief 预加载所有图像到内存中
     * 减少实时读取时的IO开销
     */
    void preloadImages();
    
    /**
     * @brief 发布当前图像
     * 定时器回调函数，负责发布图像数据
     */
    void publishImage();
    
    /**
     * @brief 发布压缩图像
     * @param image OpenCV图像数据
     * @param header ROS消息头
     */
    void publishCompressedImage(const cv::Mat& image, const std_msgs::msg::Header& header);

    /**
     * @brief 初始化节点参数
     */
    void initializeParameters();

    /**
     * @brief 创建发布者
     */
    void createPublishers();

    /**
     * @brief 打印节点信息
     */
    void printNodeInfo();

    // ROS发布者和定时器
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // 图像相关成员变量
    std::string image_folder_;                    ///< 图像文件夹路径
    std::vector<std::string> image_files_;        ///< 图像文件路径列表
    std::vector<cv::Mat> preloaded_images_;       ///< 预加载的图像数据
    size_t current_image_index_;                  ///< 当前图像索引
    
    // 配置参数
    bool loop_images_;                            ///< 是否循环播放
    bool preload_images_;                         ///< 是否预加载图像
    bool use_compression_;                        ///< 是否使用图像压缩
    int resize_width_;                            ///< 缩放宽度
    int resize_height_;                           ///< 缩放高度
    
    // 常量定义
    static const std::vector<std::string> SUPPORTED_EXTENSIONS;
};

} // namespace image_publisher

#endif // IMAGE_PUBLISHER_NODE_HPP 