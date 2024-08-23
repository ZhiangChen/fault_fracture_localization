from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="fault_fracture_localization",
            namespace="px4_1",
            executable="multi_uav_example.py",
            name="multi_uav_example",
            parameters=[
                {"drone_id":2},
                {"horizontal_offset":0}
                ]
            ),
        Node(
            package="fault_fracture_localization",
            namespace="px4_2",
            executable="multi_uav_example.py",
            name="multi_uav_example",
            parameters=[
                {"drone_id":3},
                {"horizontal_offset":2},
                ]
            ),

        Node(
            package="fault_fracture_localization",
            namespace="px4_3",
            executable="multi_uav_example.py",
            name="multi_uav_example",
            parameters=[
                {"drone_id":4},
                {"horizontal_offset":-2},
                ]
            ),
        Node(
            package="fault_fracture_localization",
            namespace="px4_4",
            executable="multi_uav_example.py",
            name="multi_uav_example",
            parameters=[
                {"drone_id":5},
                {"horizontal_offset":4},
                ]
            ),       
        Node(
            package="fault_fracture_localization",
            namespace="px4_5",
            executable="multi_uav_example.py",
            name="multi_uav_example",
            parameters=[
                {"drone_id":6},
                {"horizontal_offset":-4},
                ]
            ),
        ])
