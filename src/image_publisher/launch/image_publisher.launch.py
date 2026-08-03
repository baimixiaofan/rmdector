#!/usr/bin/env python3
"""
图像发布节点启动文件

该启动文件用于启动图像发布节点，支持多种配置参数：
- image_folder: 图片文件夹路径
- publish_rate: 发布频率
- loop: 是否循环播放
- preload_images: 是否预加载图像
- use_compression: 是否使用图像压缩
- queue_size: 发布队列大小
- resize_width/height: 图像缩放尺寸

作者: fourzkw
版本: 1.0.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.actions import OpaqueFunction

def generate_launch_description():
    """生成启动描述"""
    
    # 声明launch参数
    image_folder_arg = DeclareLaunchArgument(
        'image_folder',
        default_value='',
        description='图片文件夹路径（为空时使用默认路径）'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='10.0',
        description='图片发布频率（Hz），建议范围: 0.1-10.0'
    )
    
    loop_arg = DeclareLaunchArgument(
        'loop',
        default_value='true',
        description='是否循环播放图片（true/false）'
    )
    
    preload_images_arg = DeclareLaunchArgument(
        'preload_images',
        default_value='true',
        description='是否预加载图像到内存中（true/false）- 减少IO卡顿但增加内存使用'
    )
    
    use_compression_arg = DeclareLaunchArgument(
        'use_compression',
        default_value='false',
        description='是否使用图像压缩（true/false）- 减少网络带宽但增加CPU使用'
    )
    
    queue_size_arg = DeclareLaunchArgument(
        'queue_size',
        default_value='50',
        description='发布队列大小（建议范围: 10-100）- 增大可减少消息丢失'
    )
    
    resize_width_arg = DeclareLaunchArgument(
        'resize_width',
        default_value='0',
        description='缩放图像宽度（像素，0表示不缩放）'
    )
    
    resize_height_arg = DeclareLaunchArgument(
        'resize_height',
        default_value='0',
        description='缩放图像高度（像素，0表示不缩放）'
    )
    
    # 图像发布节点
    image_publisher_node = Node(
        package='image_publisher',
        executable='image_publisher_node',
        name='image_publisher_node',
        namespace='',
        parameters=[{
            'image_folder': LaunchConfiguration('image_folder'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'loop': LaunchConfiguration('loop'),
            'preload_images': LaunchConfiguration('preload_images'),
            'use_compression': LaunchConfiguration('use_compression'),
            'queue_size': LaunchConfiguration('queue_size'),
            'resize_width': LaunchConfiguration('resize_width'),
            'resize_height': LaunchConfiguration('resize_height')
        }],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    # 启动信息日志
    startup_log = LogInfo(
        msg=[
            '\n',
            '='*60, '\n',
            '  图像发布节点启动中...', '\n',
            '  包名称: image_publisher', '\n',
            '  节点名称: image_publisher_node', '\n',
            '  发布话题: /sensor_img (或 /sensor_img/compressed)', '\n',
            '='*60, '\n'
        ]
    )
    
    return LaunchDescription([
        # 启动参数
        image_folder_arg,
        publish_rate_arg,
        loop_arg,
        preload_images_arg,
        use_compression_arg,
        queue_size_arg,
        resize_width_arg,
        resize_height_arg,
        
        # 启动信息
        startup_log,
        
        # 节点
        image_publisher_node
    ]) 