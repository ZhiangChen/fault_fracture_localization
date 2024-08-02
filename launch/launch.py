#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="fault_fracture_localization",
            namespace="fault_fracture_localization",
            executable="offboard_control.py",
            name="offboard",
            ),

        Node(
            package="fault_fracture_localization",
            namespace="fault_fracture_localization",
            executable="perception.py",
            name="perception",
            ),

        Node(
            package="fault_fracture_localization",
            namespace="fault_fracture_localization",
            executable="state_machine.py",
            name="state_machine",
            ),
        ])

