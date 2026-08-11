#!/usr/bin/env python3
"""检测节点启动文件

启动 YOLOv8 装甲板检测节点：
- 订阅 /sensor_img（或 /sensor_img/compressed）
- 发布 /detect/image（画框图像）、/detect/detections（检测结果）和 /aim_target（瞄准目标）

参数:
- model_path: onnx 模型路径（默认 armor-4 训练好的模型）
- conf_threshold: 置信度阈值
- iou_threshold: NMS IoU 阈值
- armor_type: 装甲板图案类型（哨兵期望输出 7）
- camera_*: 相机内参（fx/fy/cx/cy）
- dist_*: 相机畸变系数（k1/k2/p1/p2/k3）
- cam_to_robot_*: 相机到机器人坐标系的外参（平移 x/y/z，旋转 roll/pitch/yaw 度）
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 注意：不能依赖 __file__ 的相对路径——launch 文件安装到 install/ 后
    # 相对路径会解析到不存在的目录，因此这里直接使用源码里的绝对路径
    # （与 detect_node 的 C++ 默认值一致，也可用 model_path:= 参数覆写）
    default_model = os.path.expanduser(
        '~/rmdector/src/detect/armor-4/weights/best.onnx')

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model,
        description='ONNX 模型路径'
    )
    conf_arg = DeclareLaunchArgument(
        'conf_threshold',
        default_value='0.25',
        description='置信度阈值'
    )
    iou_arg = DeclareLaunchArgument(
        'iou_threshold',
        default_value='0.45',
        description='NMS IoU 阈值'
    )
    save_dir_arg = DeclareLaunchArgument(
        'save_dir',
        default_value='~/rmdector/detect_results',
        description='检测结果保存文件夹（空 = 不保存）'
    )

    detect_node = Node(
        package='detect',
        executable='detect_node',
        name='detect_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'conf_threshold': LaunchConfiguration('conf_threshold'),
            'iou_threshold': LaunchConfiguration('iou_threshold'),
            'save_dir': LaunchConfiguration('save_dir'),
            # 相机标定与坐标变换参数（默认值取自 26 赛季培训说明）
            'armor_type': 7,
            'camera_fx': 1462.3697,
            'camera_fy': 1469.68385,
            'camera_cx': 398.59394,
            'camera_cy': 110.68997,
            'dist_k1': 0.003518,
            'dist_k2': -0.311778,
            'dist_p1': -0.016581,
            'dist_p2': 0.023682,
            'dist_k3': 0.0,
            'cam_to_robot_x': 0.08,
            'cam_to_robot_y': 0.0,
            'cam_to_robot_z': 0.05,
            'cam_to_robot_roll': 0.0,
            'cam_to_robot_pitch': 60.0,
            'cam_to_robot_yaw': 20.0,
        }]
    )

    startup_log = LogInfo(
        msg=[
            '\n', '=' * 60, '\n',
            '  YOLOv8 装甲板检测节点启动中...', '\n',
            '  订阅: /sensor_img (或 /sensor_img/compressed)', '\n',
            '  发布: /detect/image, /detect/detections, /aim_target', '\n',
            '=' * 60, '\n'
        ]
    )

    return LaunchDescription([
        model_path_arg,
        conf_arg,
        iou_arg,
        save_dir_arg,
        startup_log,
        detect_node,
    ])
