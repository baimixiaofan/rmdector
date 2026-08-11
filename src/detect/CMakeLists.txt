cmake_minimum_required(VERSION 3.8)
project(detect)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- 依赖 ----
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(aim_interfaces REQUIRED)

# ---- ONNX Runtime（手动安装到 ~/onnxruntime）----
set(ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime")
find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "未找到 onnxruntime 头文件，请先安装到 ${ONNXRUNTIME_ROOT}/include")
endif()

# ---- 自定义消息 ----
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  DEPENDENCIES std_msgs
)

# ---- 可执行文件 ----
add_executable(detect_node src/detect_node.cpp)

target_include_directories(detect_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ONNXRUNTIME_ROOT}/include
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(
  detect_node
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  aim_interfaces
)

target_link_libraries(detect_node ${ONNXRUNTIME_LIB} ${OpenCV_LIBS})

# 运行时查找 ~/onnxruntime/lib/libonnxruntime.so
set_target_properties(detect_node PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
  INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

# 生成自定义消息的 C++ 接口
rosidl_target_interfaces(detect_node ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ---- 安装 ----
install(TARGETS detect_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include/)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  OpenCV
  aim_interfaces
)

ament_package()
