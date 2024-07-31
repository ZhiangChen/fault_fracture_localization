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
from custom_msgs.srv._waypoint_service import WaypointService

from geometry_msgs.msg import Pose, PoseStamped
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
        Kp (float): Proportional gai]n
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

class TrajectoryGenerator:
    def __init__(self, node):
        self.waypoints = deque()
        self.current_path = []
        self.broadcast_speed = 100 # new point publishing speed in Hz
        self.node = node # TODO only for debug purposes, remove later

    def is_same(self, point1, point2):
        return round(point1.x) == round(point2.x) and round(point1.y) == round(point2.y) and round(point1.z) == round(point2.z) and round(point1.yaw) == round(point2.yaw)

    def add_waypoint(self, waypoint):
        """
        Adds a waypoint to the path

        Parameters: 
        waypoint (Waypoint): The waypoint to add to the path
        """
        self.node.get_logger().info("new waypoint recieved!")
        if len(self.waypoints) == 0 or not self.is_same(self.waypoints[-1], waypoint):
            self.waypoints.append(waypoint)

    def generate_path(self):
        """
        Generates a B-spline path which interpolates all current waypoints on the path

        Returns:
        numpy.ndarray: Array of Waypoint objects indicating the sequence of points that should be broadcast at the given Hertz,
                        None if interpolation cannot be performed
        """
        if len(self.waypoints) <= 3:
            return None

        x = [i.x for i in self.waypoints]
        y = [i.y for i in self.waypoints]
        z = [i.z for i in self.waypoints]
        yaw = [i.yaw for i in self.waypoints]
        tck, u = interpolate.splprep([x, y ,z, yaw], s = 0)
        points = np.linspace(0, 1, num=int(10000 * len(u))) 
        spline = interpolate.splev(points, tck, ext=3)
        
        #roughly calculate the length of the spline

        diff = np.diff(spline, axis=1)
        squared_distances = np.sum(diff**2, axis=0)
        distance = np.sum(np.sqrt(squared_distances))

        uav_speed = (self.waypoints[0].velocity_x ** 2 + self.waypoints[0].velocity_y ** 2 + self.waypoints[0].velocity_z ** 2) ** .5

        # TODO: hack
        if (uav_speed == 0):
            uav_speed = 10

        num_points = distance / uav_speed * self.broadcast_speed
        new_points = np.linspace(0, 1, num=int(num_points * len(u))) 
        spline = interpolate.splev(new_points, tck, ext=3)
        actual = deque()
        debug = [] #DEBUG PURPOSES ONLY
        for i in range(len(spline[0])):
            t = Waypoint()
            t.x = spline[0][i]
            t.y = spline[1][i]
            t.z = spline[2][i]
            t.yaw =  spline[3][i]

            actual.append(t)
            '''path debug
            pose = Pose()
            pose.position.x = t.x
            pose.position.y = t.y
            pose.position.z = t.z
            pose.orientation.w = 0.
            pose.orientation.x = 0.
            pose.orientation.y = 0.
            pose.orientation.z = 0.
            posestamped = PoseStamped()
            posestamped.pose = pose
            #posestamped.header.stamp = self.node.get_clock().now().nanoseconds
            posestamped.header.frame_id = "uav"
            debug.append(posestamped)
                '''

        path = Path()
        path.poses = debug
        #path.header.stamp=self.node.get_clock().now().nanoseconds
        path.header.frame_id = "uav"
        self.node.path_publisher.publish(path)

        return actual
    
    def next_point(self):
        """
        Gets the next point that should be broadcast

        Returns:
        Waypoint: Waypoint object of the next point to be broadcast, None if there are no more points
        """
        if len(self.current_path) < 1:
            new_path = self.generate_path()
            if new_path is not None:
                self.current_path = new_path
            else:
                return None
        if len(self.waypoints) > 1 and self.is_same(self.current_path[0], self.waypoints[1]):
            self.node.get_logger().info("point removed!")
            self.waypoints.popleft()
            new_path = self.generate_path()
            if new_path is not None:
                self.current_path = new_path

        next = self.current_path.popleft()
        next.velocity_x = (self.current_path[0].x - next.x) / (1 / self.broadcast_speed)
        next.velocity_y = (self.current_path[0].y - next.y) / (1 / self.broadcast_speed)
        next.velocity_z = (self.current_path[0].z - next.z) / (1 / self.broadcast_speed)
        next.velocity_yaw =  (self.current_path[0].yaw - next.yaw) / (1 / self.broadcast_speed)
        #self.node.get_logger().info("next point at " + str(next.x) + " " +  str(next.y) + " " + str(next.z))
        return next
        
        


class OffboardControl(Node):

    def __init__(self):
        super().__init__("offboard")

        qos_best_effort_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        

        odometry_callback_group = MutuallyExclusiveCallbackGroup()
        timer_callback_group = MutuallyExclusiveCallbackGroup()


        self.status_sub = self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.vehicle_status_callback, qos_best_effort_profile)
        self.offboard_mode_publisher = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.trajectory_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.vehicle_command_publisher_ = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.path_publisher = self.create_publisher(Path, "path", 10)
        #self.position_sub = self.create_subscription(VehicleLocalPosition, "/fmu/out/sensor_combined", self.debug_callback, 10)
        self.waypoint_service = self.create_service(WaypointService, "waypoint", self.waypoint_callback)
        self.uav_pose_subscriber = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.uav_pose_callback, qos_best_effort_profile, callback_group=odometry_callback_group)
        self.uav_pose_cache = []
        self.last_update = self.get_clock().now().nanoseconds
        self.uav_pose_cache_max_length = 500
        self.trajectory_generator = TrajectoryGenerator(self)
        self.distance_threshold = 5 # distance we are next to waypoint by before switching to next waypoint
        self.time = 0
        timer_period = .01
        self.timer_ = self.create_timer(timer_period, self.timer_callback, callback_group=timer_callback_group)
        self.pid_x = PID(1., 0.00, 0.01)
        self.pid_y = PID(1., 0.00, 0.01)
        self.pid_z = PID(1., 0.00, 0.01)
        self.pid_yaw = PID(2.0, 0.0, 0.0)


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
        """
        Publishes a VehicleCommand message to PX4

        Parameters:
        command: a PX4 command 
        param1: a value corresponding to the param1 field in the PX4 VehicleCommand message
        param2: a value corresponding to the param2 field in the PX4 VehicleCommand message
        """
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
        """

        """
        msg = TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [x, y, z]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        #msg.yawspeed = yaw
        self.trajectory_publisher.publish(msg)

    def uav_pose_to_string(self, uav_pose):
        """
        Generates string of uav pose for debugging purposes

        Parameters:
        uav_pose (Pose): The pose of the uav to be turned into a string

        Returns:
        String: The pose as a string
        """

        first = "POSITION: " + str(uav_pose.position.x) + " " + str(uav_pose.position.y) + " " + str(uav_pose.position.z) + " \n"
        second = "ORIENTATION " + str(uav_pose.orientation.x) + " " + str(uav_pose.orientation.y) + " " + str(uav_pose.orientation.z) + " " + str(uav_pose.orientation.w) 
        return first + second


    def uav_pose_callback(self, msg):
        """
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
        """Callback function for VehicleStatus subscription. Function currently TODO"""
        # TODO
        pass

    def waypoint_callback(self, request, response):
        """
        Acknowledges a waypoint service call and appends the waypoint to the queue 

        Parameters:
        request (Waypoint): Part of the WaypointService server message. The point to add to the UAV path
        response (Bool): Part of the WaypointService server message. Boolean acknowleging if message was recieved
        
        Returns:
        Bool: indiciation if the message was recieved
        """
        self.trajectory_generator.add_waypoint(request.waypoint)
        response.ack = True
        #self.get_logger().info("waypoint recieved and acknowleged")
        return response

    def publish_offboard_mode(self):
        """
        Publishes a PX4 OffboardControl message that indicates the UAV is flying in velocity mode
        """
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        self.offboard_mode_publisher.publish(msg)

    def timer_callback(self):
        """
        Callback for the timer and the main loop. Controls interfacing with drone movement
        """
        if (self.time == 250):
            self.get_logger().info("initing")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.arm()
            

        self.publish_offboard_mode()
        if (self.time > 1501):
            # first determine the current UAV pose
            current_time = self.get_clock().now().nanoseconds
            

            uav_pose = self.uav_pose_cache[-1][0]

            next_point = self.trajectory_generator.next_point()
            if next_point is None:
                #self.get_logger().info("No next point recieved!")
                self.publish_trajectory_command(0., 0., -.1, 0.)
            else:

                orientation = uav_pose.orientation
                position = uav_pose.position
                dt = (current_time - self.last_update) / 10000000 # to seconds

                current_heading = Rotation.from_quat([orientation.w, orientation.x, orientation.y, orientation.z]).as_euler('zyx')
                x_vel = self.pid_x.update(position.x, next_point.x, dt) + next_point.velocity_x
                y_vel = self.pid_y.update(position.y, next_point.y, dt) + next_point.velocity_y
                z_vel = self.pid_z.update(position.z + 6, next_point.z, dt) + next_point.velocity_z
                #self.get_logger().info('pose x: ' + str(uav_pose.position.x) + " next x: " + str(next_point.x))
                #self.get_logger().info('pose y: ' + str(uav_pose.position.y) + " next y: " + str(next_point.y))
                #self.get_logger().info('pose z: ' + str(uav_pose.position.z + 4) + " next z: " + str(next_point.z))
                #self.get_logger().info("current pid: " + str(x_vel) + " "+ str(y_vel) + " "+ str(z_vel) + " ")
                yaw_vel = self.pid_yaw.update(current_heading[0], next_point.yaw, dt) + next_point.velocity_yaw
                self.publish_trajectory_command(x_vel , y_vel, z_vel, yaw_vel)

            self.last_update = current_time
        elif (self.time == 1500):
            position = self.uav_pose_cache[-1][0].position
            orientation = self.uav_pose_cache[-1][0].orientation
            init = Waypoint()
            init.x = position.x
            init.y = position.y
            init.z = position.z
            heading = Rotation.from_quat([orientation.w, orientation.x, orientation.y, orientation.z]).as_euler('zyx')
            init.yaw = heading[0]
            self.trajectory_generator.add_waypoint(init)
            # testing points
            t = Waypoint()
            t.x = 50.
            t.y = 50.
            t.z = -20.
            t.yaw = 0.
            t.velocity_x = 15.
            t.velocity_y = 15.
            t.velocity_z = 15.
            t.velocity_yaw = 0.
            e = Waypoint()
            e.x = 50.
            e.y = 0.
            e.z = -40.
            e.yaw = 0.
            e.velocity_x = 8.
            e.velocity_y = 8.
            e.velocity_z = 8.
            e.velocity_yaw = 0.
            w = Waypoint()
            w.x = 25.
            w.y = 25.
            w.z = -20.
            w.yaw = 0.
            w.velocity_x = 8.
            w.velocity_y = 8.
            w.velocity_z = 8.
            w.velocity_yaw = 0.
            f = Waypoint()
            f.x = -0.
            f.y = -0.
            f.z = -20.
            f.yaw = 0.
            f.velocity_x = 8.
            f.velocity_y = 8.
            f.velocity_z = 8.
            f.velocity_yaw = 0.
            self.trajectory_generator.add_waypoint(t)
            self.trajectory_generator.add_waypoint(e)
            self.trajectory_generator.add_waypoint(w)
            self.trajectory_generator.add_waypoint(f)
        elif (self.time > 350):
                self.publish_trajectory_command(0., 0., -1., 0.)
        self.time += 1



def main(args = None):
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    executor = MultiThreadedExecutor()
    executor.add_node(offboard_control)
    executor.spin()
    offboard_control.destroy_node()
    rclpy.shutdown()



if __name__ == "__main__":
    main()


