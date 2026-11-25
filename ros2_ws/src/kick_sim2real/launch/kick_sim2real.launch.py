from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = LaunchConfiguration("params")
    return LaunchDescription([
        DeclareLaunchArgument(
            "params", default_value="config/kick_sim2real.yaml",
            description="参数文件路径"),
        Node(
            package="kick_sim2real",
            executable="policy_node",
            name="kick_sim2real",
            output="screen",
            parameters=[params],
        ),
    ])
