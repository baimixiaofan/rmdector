cmake_minimum_required(VERSION 3.8)
project(image_publisher)

# 设置C++标准
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 设置C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)

# 添加可执行文件
add_executable(image_publisher_node src/image_publisher_node.cpp)

# 设置包含目录
target_include_directories(image_publisher_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS})

# 编译特性
target_compile_features(image_publisher_node PUBLIC c_std_99 cxx_std_17)

# 链接依赖库
ament_target_dependencies(
  image_publisher_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  std_msgs
)

# 链接OpenCV库
target_link_libraries(image_publisher_node ${OpenCV_LIBS})

# 安装可执行文件
install(TARGETS image_publisher_node
  DESTINATION lib/${PROJECT_NAME})

# 安装头文件
install(DIRECTORY include/
  DESTINATION include/)

# 安装启动文件
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py")

# 安装图片目录（示例图片）
install(DIRECTORY images/
  DESTINATION share/${PROJECT_NAME}/images
  FILES_MATCHING 
  PATTERN "*.jpg"
  PATTERN "*.jpeg" 
  PATTERN "*.png"
  PATTERN "*.bmp"
  PATTERN "*.tiff"
  PATTERN "*.tif")

# 测试配置
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # 以下行跳过版权检查器
  set(ament_cmake_copyright_FOUND TRUE)
  # 以下行跳过cpplint（仅在需要时使用）
  set(ament_cmake_cpplint_FOUND TRUE)
  
  ament_lint_auto_find_test_dependencies()
endif()

# 导出依赖
ament_export_dependencies(
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  std_msgs
  OpenCV
)

ament_package() 