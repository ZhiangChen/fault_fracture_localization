#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('fault_fracture_localization'),
        'config',
        'parameters.yaml'
        )
    return LaunchDescription([
        Node(
            package="fault_fracture_localization",
            namespace="fault_fracture_localization",
            executable="offboard_control.py",
            name="offboard",
            parameters=[config],
            ),

        Node(
            package="fault_fracture_localization",
            namespace="fault_fracture_localization",
            executable="perception.py",
            name="perception",
            parameters=[config],
            ),

        Node(
            package="fault_fracture_localization",
            namespace="fault_fracture_localization",
            executable="state_machine.py",
            name="state_machine",
            parameters=[config],
            ),
        ])

