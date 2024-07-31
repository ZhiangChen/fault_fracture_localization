from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="fault_fracture_localization",
            namespace="fault_fracture_localization",
            executable="offboard_control",
            name="offboard",
            ),

        Node(
            package="fault_fracture_localization",
            namespace="fault_fracture_localization",
            executable="perception",
            name="perception",
            ),

        Node(
            package="fault_fracture_localization",
            namespace="fault_fracture_localization",
            executable="state_machine",
            name="state_machine",
            ),
        ])

