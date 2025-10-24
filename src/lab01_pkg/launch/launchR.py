from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lab01_pkg',
            executable='controller_reset',
            name='controller_reset_node',
            output='screen'
        ),
        Node(
            package='lab01_pkg',
            executable='localization_reset',
            name='localization_reset_node',
            output='screen'
        ),
        Node(
            package='lab01_pkg',
            executable='reset_node',
            name='reset_node',
            output='screen'
        )
    ])