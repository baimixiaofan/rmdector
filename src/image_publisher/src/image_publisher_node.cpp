#include "image_publisher/image_publisher_node.hpp"

/**
 * @file image_publisher_node.cpp
 * @brief ROS 2 图像发布节点实现（模拟相机）
 *
 * 作用：从指定文件夹按顺序读取图片，以固定频率发布到 ROS 话题，
 *       供下游节点（如感知、显示、录包）订阅使用。
 *
 * 发布话题：
 *   - /sensor_img            （sensor_msgs/Image，原始 BGR8 图像）
 *   - /sensor_img/compressed （sensor_msgs/CompressedImage，JPEG 压缩图像）
 *
 * 主要参数（可用 launch 文件或 --ros-args 覆写）：
 *   - image_folder    图片文件夹路径（空则用包内 images/）
 *   - publish_rate    发布频率 Hz
 *   - loop            播完是否循环；false 时发完自动退出
 *   - preload_images  是否启动时全部读入内存（省 IO、耗内存）
 *   - use_compression 是否发 JPEG 压缩图（省带宽、耗 CPU）
 *   - resize_width/height 发布前缩放尺寸（0 表示不缩放）
 */

namespace image_publisher
{

/** @brief 支持的图片文件扩展名（loadImageFiles 扫描文件夹时按此过滤） */
const std::vector<std::string> ImagePublisherNode::SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"
};

/**
 * @brief 构造函数：节点启动入口
 *
 * 整体流程：
 *   1. 读取并声明 ROS 参数（图片文件夹、发布频率、循环等）
 *   2. 按参数创建图像发布者（原始图 / 压缩图二选一）
 *   3. 扫描文件夹，收集所有支持的图片文件
 *   4. 可选：将所有图片预加载进内存（避免发布时实时读盘的 IO 卡顿）
 *   5. 创建定时器，按 publish_rate 频率周期性触发 publishImage()
 */
ImagePublisherNode::ImagePublisherNode() 
    : Node("image_publisher_node"), 
      current_image_index_(0)   ///< 当前要发布的图片下标
{
    // 初始化参数
    initializeParameters();
    
    // 创建发布者
    createPublishers();
    
    // 加载图片文件列表
    if (!loadImageFiles()) {
        RCLCPP_ERROR(this->get_logger(), "无法加载图片文件，节点将退出");
        rclcpp::shutdown();
        return;
    }
    
    // 预加载图像到内存（可选）
    if (preload_images_) {
        RCLCPP_INFO(this->get_logger(), "预加载图像到内存中...");
        preloadImages();
        RCLCPP_INFO(this->get_logger(), "图像预加载完成");
    }
    
    // 创建定时器：发布频率 -> 周期（ms），例如 10Hz -> 100ms
    double publish_rate = this->get_parameter("publish_rate").as_double();
    auto timer_period = std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate));
    timer_ = this->create_wall_timer(timer_period, std::bind(&ImagePublisherNode::publishImage, this));
    
    // 打印节点信息
    printNodeInfo();
}

/**
 * @brief 初始化节点参数
 *
 * 声明含默认值的 ROS 参数（launch 文件可通过 --ros-args 覆写），
 * 读取到成员变量，未指定图片文件夹时回退到包内默认路径
 */
void ImagePublisherNode::initializeParameters()
{
    // 声明参数（含默认值），launch 文件可通过 --ros-args 参数覆写
    this->declare_parameter("image_folder", "");
    this->declare_parameter("publish_rate", 1.0);
    this->declare_parameter("loop", false);  // 默认不循环播放
    this->declare_parameter("preload_images", true);  // 默认预加载，确保每张图片只读取一次
    this->declare_parameter("use_compression", false);
    this->declare_parameter("queue_size", 50);
    this->declare_parameter("resize_width", 0);
    this->declare_parameter("resize_height", 0);
    
    // 获取参数
    image_folder_ = this->get_parameter("image_folder").as_string(); #这里的.as_string()是获取参数的值，并将其转换为字符串类型
    loop_images_ = this->get_parameter("loop").as_bool();
    preload_images_ = this->get_parameter("preload_images").as_bool();
    use_compression_ = this->get_parameter("use_compression").as_bool();
    resize_width_ = this->get_parameter("resize_width").as_int();
    resize_height_ = this->get_parameter("resize_height").as_int();
    
    // 如果没有指定图片文件夹，使用默认路径
    if (image_folder_.empty()) {
        // 尝试使用包内的images文件夹
        image_folder_ = std::filesystem::current_path() / "src/image_publisher/images";
        RCLCPP_INFO(this->get_logger(), "使用默认图片文件夹: %s", image_folder_.c_str());
    }
}

/**
 * @brief 创建发布者
 *
 * 根据 use_compression_ 创建原始图像或压缩图像发布者，二选一；
 * 队列深度取 queue_size 参数
 */
void ImagePublisherNode::createPublishers()
{
    int queue_size = this->get_parameter("queue_size").as_int();
    
    // 创建发布者 - 使用更大的队列深度
    // 注意：原始图和压缩图只会创建其中一个
    if (use_compression_) {
        // 压缩模式：JPEG 压缩后的图像，带宽小、CPU 占用高
        compressed_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
            "/sensor_img/compressed", queue_size);
        RCLCPP_INFO(this->get_logger(), "使用压缩图像发布");
    } else {
        // 原始模式：BGR8 原始图像，数据量大但无需解压
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/sensor_img", queue_size);
        RCLCPP_INFO(this->get_logger(), "使用原始图像发布");
    }
}

/**
 * @brief 打印节点信息
 *
 * 启动时输出图片文件夹、文件数量、发布频率、循环/预加载等配置，
 * 还会就是否循环播放、是否预加载、缩放尺寸等参数，
 * 便于确认运行参数是否符合预期
 */
void ImagePublisherNode::printNodeInfo()
{
    double publish_rate = this->get_parameter("publish_rate").as_double();
    int queue_size = this->get_parameter("queue_size").as_int();
    
    RCLCPP_INFO(this->get_logger(), "图像发布节点已启动");
    RCLCPP_INFO(this->get_logger(), "图片文件夹: %s", image_folder_.c_str());
    RCLCPP_INFO(this->get_logger(), "找到 %zu 个图片文件", image_files_.size());
    RCLCPP_INFO(this->get_logger(), "发布频率: %.1f Hz", publish_rate);
    RCLCPP_INFO(this->get_logger(), "循环播放: %s", loop_images_ ? "是" : "否");
    RCLCPP_INFO(this->get_logger(), "预加载图像: %s", preload_images_ ? "是（每张图片只读取一次）" : "否");
    RCLCPP_INFO(this->get_logger(), "队列大小: %d", queue_size);
    
    if (!loop_images_) {
        RCLCPP_INFO(this->get_logger(), "注意：循环播放已禁用，所有图片发布完毕后节点将自动退出");
    }
    if (resize_width_ > 0 && resize_height_ > 0) {
        RCLCPP_INFO(this->get_logger(), "图像缩放至: %dx%d", resize_width_, resize_height_);
    }
}

/**
 * @brief 加载图片文件列表
 *
 * 检查文件夹是否存在，遍历收集扩展名匹配（见 SUPPORTED_EXTENSIONS）
 * 的普通文件并按文件名排序，保证发布顺序与文件名一致
 * @return 成功返回 true，失败返回 false
 */
bool ImagePublisherNode::loadImageFiles()
{
    // 检查文件夹是否存在
    if (!std::filesystem::exists(image_folder_)) {
        RCLCPP_ERROR(this->get_logger(), "图片文件夹不存在: %s", image_folder_.c_str());
        return false;
    }
    
    try {
        // 遍历文件夹，收集所有扩展名匹配的普通文件（支持格式见 SUPPORTED_EXTENSIONS）
        for (const auto& entry : std::filesystem::directory_iterator(image_folder_)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);  #转小写工程化的代码。。
                
                if (std::find(SUPPORTED_EXTENSIONS.begin(), SUPPORTED_EXTENSIONS.end(), extension) 
                    != SUPPORTED_EXTENSIONS.end()) {
                    image_files_.push_back(entry.path().string());
                }
            }
        }
    } 
    catch (const std::filesystem::filesystem_error& ex) {
        RCLCPP_ERROR(this->get_logger(), "读取文件夹时出错: %s", ex.what());
        return false;
    }
    
    // 文件夹内没有支持的图片则视为加载失败
    if (image_files_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "在文件夹 %s 中没有找到支持的图片文件", image_folder_.c_str());
        return false;
    }
    
    // 对文件名进行排序（保证发布顺序与文件名一致）
    std::sort(image_files_.begin(), image_files_.end());
    return true;
}

/**
 * @brief 预加载所有图像到内存中
 *
 * 一次性把全部图片读入内存（按需缩放），之后发布时不再访问磁盘。
 * 无法读取的图片放入空 cv::Mat 占位，保持下标与 image_files_ 一一对应
 */
void ImagePublisherNode::preloadImages()
{
    // 一次性把全部图片读入内存，之后发布时不再访问磁盘
    preloaded_images_.reserve(image_files_.size());
    
    for (const auto& image_path : image_files_) {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (!image.empty()) {
            // 如果设置了缩放参数，则缩放图像
            if (resize_width_ > 0 && resize_height_ > 0) {
                cv::Mat resized_image;
                cv::resize(image, resized_image, cv::Size(resize_width_, resize_height_));
                preloaded_images_.push_back(resized_image);
            } else {
                preloaded_images_.push_back(image);
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "无法预加载图片: %s", image_path.c_str());
            // 添加空图像占位符，保持下标与 image_files_ 一一对应
            preloaded_images_.push_back(cv::Mat());
        }
    }
}

/**
 * @brief 发布当前图像
 *
 * 定时器回调：索引越界时按 loop_images_ 决定回到第一张或取消定时器、
 * 延迟 100ms 后关闭节点（确保最后一条消息已发出）；图像优先取预加载的
 * 内存图像，否则实时读盘，随后按 use_compression_ 走压缩或原始图发布
 */
void ImagePublisherNode::publishImage()
{
    if (image_files_.empty()) {
        return;
    }
    
    // 检查是否需要循环：所有图片已发完时
    if (current_image_index_ >= image_files_.size()) {
        if (loop_images_) {
            current_image_index_ = 0;   // 回到第一张，重新开始
            RCLCPP_INFO(this->get_logger(), "重新开始循环播放图片");
        } else {
            // 不循环：取消定时器，延迟 100ms 后关闭节点
            // （延迟是为了确保最后一条消息已从发送队列发出）
            RCLCPP_INFO(this->get_logger(), "所有图片已发布完毕（共%zu张），节点即将退出", image_files_.size());
            timer_->cancel();
            // 延迟一小段时间后退出，确保最后的消息能被发送
            auto exit_timer = this->create_wall_timer(
                std::chrono::milliseconds(100),
                [this]() { rclcpp::shutdown(); }
            );
            return;
        }
    }
    
    cv::Mat image;
    
    // 获取图像数据：优先取预加载的内存图像，否则实时读盘
    if (preload_images_ && current_image_index_ < preloaded_images_.size()) {
        image = preloaded_images_[current_image_index_];
    } else {
        // 实时读取图片
        std::string current_image_path = image_files_[current_image_index_];
        image = cv::imread(current_image_path, cv::IMREAD_COLOR);
        
        // 如果设置了缩放参数，则缩放图像
        if (!image.empty() && resize_width_ > 0 && resize_height_ > 0) {
            cv::Mat resized_image;
            cv::resize(image, resized_image, cv::Size(resize_width_, resize_height_));
            image = resized_image;
        }
    }
    
    if (image.empty()) {
        // 读取失败：跳过这张，继续下一张
        RCLCPP_WARN(this->get_logger(), "无法读取图片: %s", 
                   image_files_[current_image_index_].c_str());
        current_image_index_++;
        return;
    }
    
    // 转换为ROS图像消息
    try {
        std_msgs::msg::Header header;
        header.stamp = this->now();          // 打上当前时间戳
        header.frame_id = "camera_frame";    // 坐标帧（模拟相机）
        
        if (use_compression_) {
            // 发布压缩图像
            publishCompressedImage(image, header);
        } else {
            // 发布原始图像：cv::Mat 转 sensor_msgs/Image（bgr8 编码）
            cv_bridge::CvImage cv_image(header, "bgr8", image);
            sensor_msgs::msg::Image::SharedPtr msg = cv_image.toImageMsg();
            publisher_->publish(*msg);
        }
        
        RCLCPP_INFO(this->get_logger(), "发布图片: %s [%zu/%zu]", 
                    std::filesystem::path(image_files_[current_image_index_]).filename().c_str(),
                    current_image_index_ + 1, image_files_.size());
        
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge异常: %s", e.what());
    }
    
    current_image_index_++;   // 指向下一张图片
}

/**
 * @brief 发布压缩图像
 *
 * 用 OpenCV 将 cv::Mat 编码为 JPEG 字节流（质量 90%），
 * 封装为 CompressedImage 消息发布到压缩话题
 * @param image  OpenCV 图像数据
 * @param header ROS 消息头
 */
void ImagePublisherNode::publishCompressedImage(const cv::Mat& image, const std_msgs::msg::Header& header)
{
    auto compressed_msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
    compressed_msg->header = header;
    compressed_msg->format = "jpeg";   // 压缩编码格式
    
    // 压缩图像：OpenCV 将 cv::Mat 编码为 JPEG 字节流
    std::vector<uchar> buffer;
    std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 90}; // 90%质量
    cv::imencode(".jpg", image, buffer, compression_params);
    
    compressed_msg->data = buffer;
    compressed_publisher_->publish(std::move(compressed_msg));
}

} // namespace image_publisher

/**
 * @brief 程序入口
 *
 * 初始化 ROS → 创建节点 → 进入事件循环（定时器回调在此触发）→ 退出时清理
 * @param argc 命令行参数个数
 * @param argv 命令行参数
 * @return 退出码
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<image_publisher::ImagePublisherNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} 