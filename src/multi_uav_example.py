#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import VehicleLocalPosition
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class OffboardControl(Node):

    def __init__(self):
        super().__init__("multi_uav")
        self.declare_parameters(
            namespace='',
            parameters=[
                ('drone_id', rclpy.Parameter.Type.INTEGER),
                ('horizontal_offset', rclpy.Parameter.Type.INTEGER)
            ])
        qos_best_effort_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        #self.status_sub = self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.vehicle_status_callback, qos_profile)
        self.offboard_mode_publisher = self.create_publisher(OffboardControlMode, "fmu/in/offboard_control_mode", 10)
        self.trajectory_publisher = self.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', 10)
        self.vehicle_command_publisher_ = self.create_publisher(VehicleCommand, "fmu/in/vehicle_command", 10)
        self.odometry_sub = self.create_subscription(VehicleOdometry, "fmu/out/vehicle_odometry", self.pose_callback, qos_best_effort_profile)
        self.time = 0
        timer_period = .1
        self.timer_ = self.create_timer(timer_period, self.cmdloop_callback)
        self.uav_pose = None
        self.start_pose = None
        self.drone_id = self.get_parameter("drone_id").value
        self.horizontal_offset = self.get_parameter("horizontal_offset").value

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, target_system=self.drone_id)
        self.get_logger().info("Arm command sent")

    def disarm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self.get_logger().info("Disarm command sent")

    def pose_callback(self, msg):
        """
        Callback function for the UAV pose subscriber
        """
        self.uav_pose = msg

    def publish_vehicle_command(self, command, param1 = 0.0, param2=0.0, target_system=1):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = target_system  # system which should execute the command
        msg.target_component = 1  # component which should execute the command, 0 for all components
        msg.source_system = 1  # system sending the command
        msg.source_component = 1  # component sending the command
        msg.from_external = True
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.vehicle_command_publisher_.publish(msg)

    def publish_trajectory_command(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.position = [x, y, z]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.yaw = float('nan')
        self.trajectory_publisher.publish(msg)

    def debug_callback(self, data):
        self.get_logger().info(str(data.z) + " " + str(data.vz))


    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        self.offboard_mode_publisher.publish(msg)


    def cmdloop_callback(self):
        if (self.uav_pose is None):
            return

        self.time += 1
        self.publish_offboard_mode()
        if (self.time == 10):
            self.get_logger().info("initing")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0, self.drone_id)
            self.arm()
            self.start_pose = self.uav_pose.position
        if (self.time < 200):
            if (self.start_pose is None):
                return
            self.publish_trajectory_command(float(self.horizontal_offset), 0., -10.)
        else:
            self.publish_trajectory_command(float(self.horizontal_offset), 400., -10.) 

def main(args = None):
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

