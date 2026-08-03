#!/usr/bin/env python3
"""检测节点启动文件

启动 YOLOv8 装甲板检测节点：
- 订阅 /sensor_img（或 /sensor_img/compressed）
- 发布 /detect/image（画框图像）和 /detect/detections（检测结果）

参数:
- model_path: onnx 模型路径（默认 armor-4 训练好的模型）
- conf_threshold: 置信度阈值
- iou_threshold: NMS IoU 阈值
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_model = os.path.join(
        os.path.dirname(__file__), '..', 'armor-4', 'weights', 'best.onnx')
    default_model = os.path.abspath(default_model)

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
        }]
    )

    startup_log = LogInfo(
        msg=[
            '\n', '=' * 60, '\n',
            '  YOLOv8 装甲板检测节点启动中...', '\n',
            '  订阅: /sensor_img (或 /sensor_img/compressed)', '\n',
            '  发布: /detect/image, /detect/detections', '\n',
            '=' * 60, '\n'
        ]
    )

    return LaunchDescription([
        model_path_arg,
        conf_arg,
        iou_arg,
        startup_log,
        detect_node,
    ])
