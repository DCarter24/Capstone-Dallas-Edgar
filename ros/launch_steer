from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rplidar_ros',
            executable='steering_control',
            name='steering_control_node',
            output='screen'
        ),
        Node(
            package='rplidar_ros',
            executable='lidar_object_detection',
            name='lidar_object_detection_node',
            output='screen'
        ),
    ])
