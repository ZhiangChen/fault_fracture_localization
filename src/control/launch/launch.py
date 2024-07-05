from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="control",
            namespace="control",
            executable="offboard_control",
            name="offboard",
            ),

        Node(
            package="control",
            namespace="control",
            executable="perception",
            name="perception",
            ),

        Node(
            package="control",
            namespace="control",
            executable="optimization",
            name="optimization",
            ),

        ])

