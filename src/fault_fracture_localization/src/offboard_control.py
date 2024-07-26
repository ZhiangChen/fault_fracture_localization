import rclpy
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import interpolate
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from collections import deque

from geometry_msgs.msg import Pose
from nav_msgs.msg import Path
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand, VehicleStatus, VehicleOdometry
from custom_msgs.msg._waypoint import Waypoint

class PID:
    def __init__(self, Kp, Ki, Kd, integral_window_size=500):
        """
        Initialize the PID controller with given gains and integral window size.
        
        Parameters:
        Kp (float): Proportional gain
        Ki (float): Integral gain
        Kd (float): Derivative gain
        integral_window_size (float): Number of recent errors to consider for the integral term
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_errors = deque(maxlen=integral_window_size)
        self.previous_error = 0

    def update(self, measurement, setpoint, dt):
        """
        Update the PID controller with the current measurement and setpoint.

        Parameters:
        measurement (float): Current measurement
        setpoint (float): Desired setpoint value
        dt (float): Time interval since the last update

        Returns:
        return (float): Control output
        """
        error = setpoint - measurement
        self.integral_errors.append(error * dt)
        integral = sum(self.integral_errors)
        derivative = (error - self.previous_error) / dt
        self.previous_error = error

        output = self.Kp * error + self.Ki * integral + self.Kd * derivative
        return output

    def clear(self):
        """
        Resets the integral and derivative part of the PID 
        """
        self.integral_errors.clear()
        self.previous_error = 0

class Path:
    def __init__(self, node):
        self.waypoints = deque()
        self.last_path = []
        self.node = node # TODO only for debug purposes, remove later


    def is_same(self, point1, point2):
        return point1.x == point2.y and point1.y == point2.y and point1.z == point2.z and point1.yaw == point2.yaw

    def add_waypoint(self, waypoint):
        if len(self.waypoints) == 0 or not self.is_same(self.waypoints[-1], waypoint):
            self.waypoints.append(waypoint)
            self.node.get_logger().info("successfully added waypoint!")
        new_path = self.generate_path()
        if new_path is not None:
            self.last_path = new_path

    def prune_waypoints(self, pose, waypoints):
        position = pose.position
        while len(self.waypoints) > 1:
            uav_to_second = (waypoints[0].x - position.x) ** 2 + (waypoints[0].y - position.y) ** 2 + (waypoints[0].z - position.z) ** 2
            first_to_second = (waypoints[1].x - waypoints[0].x) ** 2 + (waypoints[1].y - waypoints[0].y) ** 2 + (waypoints[1].z - waypoints[0].z) ** 2
            if uav_to_second < first_to_second:
                self.waypoints.popleft()
                self.node.get_logger().info("point pruned")
            else:
                break


    def generate_path(self):
        if len(self.waypoints) <= 3:
            return None

        x = [i.x for i in self.waypoints]
        y = [i.y for i in self.waypoints]
        z = [i.z for i in self.waypoints]
        yaw = [i.yaw for i in self.waypoints]
        tck, u = interpolate.splprep([x, y ,z, yaw], s = 0)
        xx = np.linspace(0, 1, num=int(10 * len(u)))
        spline = interpolate.splev(xx, tck, ext=3)
        actual = []
        for i in range(len(spline[0])):
            t = Waypoint()
            t.x = spline[0][i]
            t.y = spline[1][i]
            t.z = spline[2][i]
            t.yaw =  spline[3][i]
            actual.append(t)
        return actual
    
    def next_objective(self, pose):
        self.prune_waypoints(pose, self.last_path)
        self.prune_waypoints(pose, self.waypoints)
        if len(self.last_path) < 1:
            return None
        else:
            return self.last_path[0]


class OffboardControl(Node):

    def __init__(self):
        super().__init__("offboard")

        qos_best_effort_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)


        self.status_sub = self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.vehicle_status_callback, qos_best_effort_profile)
        self.offboard_mode_publisher = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.trajectory_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.vehicle_command_publisher_ = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        #self.position_sub = self.create_subscription(VehicleLocalPosition, "/fmu/out/sensor_combined", self.debug_callback, 10)
        self.waypoint_subscriber = self.create_subscription(Waypoint, "waypoint", self.waypoint_callback, 10)
        self.uav_pose_subscriber = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.uav_pose_callback, qos_best_effort_profile)
        self.uav_pose_cache = []
        self.last_update = self.get_clock().now().nanoseconds
        self.uav_pose_cache_max_length = 500
        self.path = Path(self)
        self.distance_threshold = 5 # distance we are next to waypoint by before switching to next waypoing
        self.time = 0
        timer_period = .1
        self.timer_ = self.create_timer(timer_period, self.cmdloop_callback)
        self.pid_x = PID(2.0, 0.1, 0.0)
        self.pid_y = PID(0.8, 0.1, 0.0)
        self.pid_z = PID(0.8, 0.1, 0.0)
        self.pid_yaw = PID(2.0, 0.1, 0.0)


    def arm(self):
        """
        Sends arm command to vehicle
        """
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command sent")

    def disarm(self):
        """
        Sends disarm command to vehicle
        """
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self.get_logger().info("Disarm command sent")

    def publish_vehicle_command(self, command, param1 = 0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1  # system which should execute the command
        msg.target_component = 1  # component which should execute the command, 0 for all components
        msg.source_system = 1  # system sending the command
        msg.source_component = 1  # component sending the command
        msg.from_external = True
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.vehicle_command_publisher_.publish(msg)

    def publish_trajectory_command(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [x, y, z]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.yawspeed = yaw
        self.trajectory_publisher.publish(msg)

    def uav_pose_callback(self, msg):
        """        self.pid_x = PID(2.0, 0.1, 0.0)
        self.pid_y = PID(0.8, 0.1, 0.0)
        self.pid_z = PID(0.8, 0.1, 0.0)
        self.pid_yaw = PID(2.0, 0.1, 0.0)
        Callback function for VehicleOdometry subscription. Adds current pose and timestamp
        to cache.

        Parameters:
        msg (VehicleOdometry): A VehicleOdometry message published by PX4

        """
        uav_pose = Pose()
        uav_pose.position.x = float(msg.position[0])
        uav_pose.position.y = float(msg.position[1])
        uav_pose.position.z = float(msg.position[2])
        uav_pose.orientation.x = float(msg.q[0])
        uav_pose.orientation.y = float(msg.q[1])
        uav_pose.orientation.z = float(msg.q[2])
        uav_pose.orientation.w = float(msg.q[3])
        #self.get_logger().info(self.uav_pose_to_string(uav_pose))
        self.uav_pose_cache.append((uav_pose, msg.timestamp))
        
        if len(self.uav_pose_cache) > self.uav_pose_cache_max_length:
            del self.uav_pose_cache[0]

    def vehicle_status_callback(self, data):
        # TODO
        pass

    def waypoint_callback(self, data):
        #self.get_logger().info(str(data.x))
        self.path.add_waypoint(data)

    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        self.offboard_mode_publisher.publish(msg)


    def generate_trajectory(self):
        # first determine the current UAV pose
        current_time = self.get_clock().now().nanoseconds
        if (len(self.uav_pose_cache) < 1):
            return

        l = -1
        r = len(self.uav_pose_cache)
        while (r - l > 1):
            m = int((l + r) / 2)
            if (current_time < self.uav_pose_cache[m][1]):
                r = m
            else:
                l = m

        uav_pose = self.uav_pose_cache[l][0]

        next_point = self.path.next_objective(uav_pose)
        if next_point is None:
            #self.get_logger().info("No next point recieved!")
            return None

        orientation = uav_pose.orientation
        position = uav_pose.position
        dt = current_time - self.last_update
        debug_str = 
        self.get_logger().info("point recieved at")

        desired_heading = Rotation.from_quat([orientation.w, orientation.x, orientation.y, orientation.z]).as_euler('zyx')
        x_vel = self.pid_x.update(position.x, next_point.x, dt)
        y_vel = self.pid_y.update(position.y, next_point.y, dt)
        z_vel = self.pid_z.update(position.z, next_point.z, dt)
        yaw_vel = self.pid_yaw.update(desired_heading[0], next_point.yaw, dt)

        self.last_update = current_time

        return [x_vel, y_vel, z_vel, yaw_vel]


    def cmdloop_callback(self):
        if (self.time == 30):
            self.get_logger().info("initing")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.arm()
            position = self.uav_pose_cache[-1][0].position
            orientation = self.uav_pose_cache[-1][0].orientation
            init = Waypoint()
            init.x = position.x
            init.y = position.y
            init.z = position.z
            heading = Rotation.from_quat([orientation.w, orientation.x, orientation.y, orientation.z]).as_euler('zyx')
            init.yaw = heading[0]
            self.path.add_waypoint(init)
            # testing
            t = Waypoint()
            t.x = 100.
            t.y = 100.
            t.z = -20.
            t.yaw = 0.
            e = Waypoint()
            e.x = 100.
            e.y = 150.
            e.z = -20.
            e.yaw = 0.
            w = Waypoint()
            w.x = 150.
            w.y = 150.
            w.z = -20.
            w.yaw = 0.
            self.path.add_waypoint(t)
            self.path.add_waypoint(e)
            self.path.add_waypoint(w)


        self.publish_offboard_mode()
        if (self.time > 20):
            trajectory = self.generate_trajectory()
            if (trajectory is not None):
                self.publish_trajectory_command(trajectory[0], trajectory[1], trajectory[2], trajectory[3])
            else:
                self.publish_trajectory_command(0., 0., -.1, 0.)
        self.time += 1



def main(args = None):
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


