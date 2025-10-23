from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lab01_pkg',
            executable='controller',
            name='controller_node',
            output='screen'
        ),
        Node(
            package='lab01_pkg',
            executable='localization',
            name='localization_node',
            output='screen'
        ),
        Node(
            package='lab01_pkg',
            executable='reset_node',
            name='reset_node',
            output='screen'
        )
    ])