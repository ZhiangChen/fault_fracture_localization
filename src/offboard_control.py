#!/usr/bin/env python3
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
from fault_fracture_localization.srv._waypoint_service import WaypointService
from fault_fracture_localization.msg._waypoint import Waypoint
from geometry_msgs.msg import Pose
from nav_msgs.msg import Path
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand, VehicleStatus, VehicleOdometry
from std_msgs.msg import String, Bool

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


class OffboardControl(Node):

    def __init__(self):
        super().__init__("offboard")

        self.declare_parameters(
            namespace = '',
            parameters = [
                    ("control_rate", rclpy.Parameter.Type.INTEGER),
                    ("target_accel", rclpy.Parameter.Type.DOUBLE),
                    ("time_upper_bound", rclpy.Parameter.Type.DOUBLE),
                    ("time_lower_bound", rclpy.Parameter.Type.DOUBLE),
                ]
        )

        # QOS profiles
        qos_best_effort_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        
        # Callback groups
        odometry_callback_group = MutuallyExclusiveCallbackGroup()
        timer_callback_group = MutuallyExclusiveCallbackGroup()

        # Subscriptions
        self.status_sub = self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.vehicle_status_callback, qos_best_effort_profile)
        self.uav_pose_subscriber = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.uav_pose_callback, qos_best_effort_profile, callback_group=odometry_callback_group)

        # Publishers
        self.offboard_mode_publisher = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
        self.trajectory_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.vehicle_command_publisher_ = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        #self.path_publisher = self.create_publisher(Path, "path", 10)
        self.path_status_publisher = self.create_publisher(Bool, "path_status", 10)

        # Services
        self.waypoint_service = self.create_service(WaypointService, "waypoints", self.waypoint_callback)

        # PID
        self.pid_x = PID(2., 0.08, 0.01)
        self.pid_y = PID(2., 0.08, 0.01)
        self.pid_z = PID(2., 0.00, 0.01)
        self.pid_yaw = PID(2.0, 0.0, 0.0)

        # timer callback
        self.control_rate = self.get_parameter("control_rate").value
        self.sampling_duration = 1.0 / self.control_rate
        self.timer_ = self.create_timer(self.sampling_duration, self.timer_callback, callback_group=timer_callback_group)
        
        # Variables
        self.uav_pose = None
        self.path = []
        self.path_velocity = []
        self.mode = "init"
        self.arm_status = False
        self.min_accel = self.get_parameter("target_accel").value # Max acceleration target for drone
        self.time_lower_bound = self.get_parameter("time_lower_bound").value # Minimum expected time for the drone to take from waypoint to another
        self.time_upper_bound = self.get_parameter("time_upper_bound").value # Maximum exptected time for drone to take from one waypoint to another

        
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

    def publish_offboard_mode(self, mode):
        """
        Publishes a PX4 OffboardControl message that indicates the mode the UAV is flying in

        Parameters:
        mode (string): the name of the mode that the UAV will be flying in
        """
        msg = OffboardControlMode()
        if (mode == "velocity"):
            msg.position = False
            msg.velocity = True
            msg.acceleration = False
        if (mode == "position"):
            msg.position = True
            msg.velocity = False
            msg.acceleration = False
        self.offboard_mode_publisher.publish(msg)

    def vehicle_status_callback(self, msg):
        """
        Callback function for the vehicle status subscriber
        """
        if msg.arming_state == 2:
            self.arm_status = True
        else:
            self.arm_status = False

    def uav_pose_callback(self, msg):
        """
        Callback function for the UAV pose subscriber
        """
        self.uav_pose = msg

    def waypoint_callback(self, request, response):
        """
        Callback function for the waypoint service
        """
        if (request.action == "takeoff"):
            self.mode = "takeoff"
            self.path = [[request.waypoints[0].x, request.waypoints[0].y, request.waypoints[0].z, request.waypoints[0].yaw]]
            self.get_logger().info("Received position request")
            response.ack = True
        elif (request.action == "position"):
            self.mode = "position"
            self.path = [[request.waypoints[0].x, request.waypoints[0].y, request.waypoints[0].z, request.waypoints[0].yaw]]
            self.get_logger().info("Received position request")
            response.ack = True
        elif (request.action == "waypoints"):
            self.mode = "waypoints"
            self.get_logger().info("Received path request")
            waypoints = request.waypoints
            # TODO: if two sequential waypoints are the same, response false
            self.path, self.path_velocity = self.generate_path_cubic(waypoints)
            self.get_logger().info("Path generated")
            self.get_logger().info(str(len(self.path)))
            response.ack = True
        else:
            self.get_logger().info("Invalid request")
            response.ack = False

        return response

    def cubic_solver(self, initial_position, initial_velocity, final_position, final_velocity, time):
        """
        Finds the appropriate cubic fit for given parameters on a certain axis

        Parameters:
        initial_position (float): The initial position of the UAV
        initial_velocity (float): The initial velocity of the UAV
        final_position (float): The final position of the UAV
        final_velocity (float): The final velocity of the UAV

        Returns:
        np.ndarray: A list of parameters in the order of a, b, c, d where ax^3 + bx^2 + cx + d fits the conditions
        """
        matrix = np.array([[0,0,0,1],
                           [0,0,1,0],
                           [time ** 3, time ** 2, time, 1],
                           [3 * (time ** 2), 2 * time, 1 , 0]])
        knowns = np.array([initial_position, initial_velocity, final_position, final_velocity])
        params = np.matmul(np.linalg.inv(matrix), knowns)
        #self.get_logger().info("inputs: " + str(initial_position) + " " + str(initial_velocity) + " " + str(final_position) + " " + str(final_velocity) + " " + str(time))
        #self.get_logger().info("params: " + str(params[0]) + " " + str(params[1]) + " " + str(params[2]) + " " + str(params[3]))
        return params

    def determine_traj_time(self, initial_position, initial_velocity, final_position, final_velocity, max_iter = 200):
        """
        Determines the least amount of time needed such that the maximum acceleration using the given parameters is the least greater than the acceleration specified

        Parameters:
        initial_position (float): The initial position of the UAV
        initial_velocity (float): The initial velocity of the UAV
        final_position (float): The final position of the UAV
        final_velocity (float): The final velocity of the UAV
        max_iter (int): The maximum number of iterations the binary search will go through
        """
        l = self.time_lower_bound - 1
        r = self.time_upper_bound + 1
        iter = 0
        while (True):
            iter += 1
            m = (l + r) / 2.0
            params = self.cubic_solver(initial_position, initial_velocity, final_position, final_velocity, m)
            #self.get_logger().info(str(6 * params[0] * m + 2 * params[1]))
            #self.get_logger().info(str(2 * params[1]))
            max_accel = max(abs(6 * params[0] * m + 2 * params[1]), abs(2 * params[1]))
            if (max_accel > self.min_accel):
                l = m
            else:
                r = m
            if abs(max_accel - self.min_accel) < .1: 
                break
            elif iter > max_iter:
                # self.get_logger().info("Did not find suitable time period in given iterations!")
                # self.get_logger().info("current_accel: " + str(max_accel) + " target accel: " + str(self.min_accel))
                break
        return r


    def generate_path_cubic(self, waypoints):
        """
        Generate a cubic based piecewise interpolation of the waypoints

        Parameters:
        waypoints (Waypoint[]): List of waypoints to be interpolated

        Returns:
        np.ndarray: Array of 4-tuples detailing the x, y, z, and yaw positions along the path. Number of points determined by control_rate
        np.ndarray: Array of 4-tuples detailing the x, y, z, and yaw velocities along the path. Number of points determined by control_rate
        """
        path = []
        path_velocity = []
        for i in range(len(waypoints) - 1):
            x_time = self.determine_traj_time(waypoints[i].x, waypoints[i].velocity_x, waypoints[i + 1].x, waypoints[i + 1].velocity_x)
            y_time = self.determine_traj_time(waypoints[i].y, waypoints[i].velocity_y, waypoints[i + 1].y, waypoints[i + 1].velocity_y)
            z_time = self.determine_traj_time(waypoints[i].z, waypoints[i].velocity_z, waypoints[i + 1].z, waypoints[i + 1].velocity_z)
            yaw_time = self.determine_traj_time(waypoints[i].yaw, waypoints[i].velocity_yaw, waypoints[i + 1].yaw, waypoints[i + 1].velocity_yaw)
            time = max(x_time, y_time, z_time, yaw_time)
            x_param = self.cubic_solver(waypoints[i].x, waypoints[i].velocity_x, waypoints[i + 1].x, waypoints[i + 1].velocity_x, time)
            y_param = self.cubic_solver(waypoints[i].y, waypoints[i].velocity_y, waypoints[i + 1].y, waypoints[i + 1].velocity_y, time)
            z_param = self.cubic_solver(waypoints[i].z, waypoints[i].velocity_z, waypoints[i + 1].z, waypoints[i + 1].velocity_z, time)
            yaw_param = self.cubic_solver(waypoints[i].yaw, waypoints[i].velocity_yaw, waypoints[i + 1].yaw, waypoints[i + 1].velocity_yaw, time)
            mesh = np.linspace(0, time, int(time * self.control_rate)) 
            x_pos = x_param[0] * mesh ** 3 + x_param[1] * mesh ** 2 + x_param[2] * mesh + x_param[3]
            y_pos = y_param[0] * mesh ** 3 + y_param[1] * mesh ** 2 + y_param[2] * mesh + y_param[3]
            z_pos = z_param[0] * mesh ** 3 + z_param[1] * mesh ** 2 + z_param[2] * mesh + z_param[3]
            yaw_pos = yaw_param[0] * mesh ** 3 + yaw_param[1] * mesh ** 2 + yaw_param[2] * mesh + yaw_param[3]

            x_vel = x_param[0] * 3 * mesh ** 2 + x_param[1] * 2 * mesh + x_param[2] 
            y_vel = y_param[0] * 3 * mesh ** 2 + y_param[1] * 2 * mesh + y_param[2] 
            z_vel = z_param[0] * 3 * mesh ** 2 + z_param[1] * 2 * mesh + z_param[2] 
            yaw_vel = yaw_param[0] * 3 * mesh ** 2 + yaw_param[1] * 2 * mesh + yaw_param[2] 

            path.extend([(x_pos[i], y_pos[i], z_pos[i], yaw_pos[i]) for i in range(len(x_pos))])  
            path_velocity.extend([(x_vel[i], y_vel[i], z_vel[i], yaw_vel[i]) for i in range(len(x_vel))])


        return path, path_velocity


    def generate_path_bspline(self, waypoints):
        """
        Generates a B-spline path which interpolates all current waypoints on the path

        Returns:
        numpy.ndarray: Array of Waypoint objects indicating the sequence of points that should be broadcast at the given Hertz,
                        None if interpolation cannot be performed
        """
        if len(waypoints) != 4:
            return None
        
        x = [i.x for i in waypoints]
        y = [i.y for i in waypoints]
        z = [i.z for i in waypoints]
        yaw = [i.yaw for i in waypoints]
        self.get_logger().info(str(len(x)))
        tck, u = interpolate.splprep([x, y , z, yaw], s = 0)
        # Interpolating the spline
        sampling_points = np.linspace(0, 1, num=int(self.control_rate * len(u)))
        spline = interpolate.splev(sampling_points, tck, ext=3)

        # calculate the velocity of the UAV at each point in tau space
        velocity_x = np.gradient(spline[0], sampling_points)
        velocity_y = np.gradient(spline[1], sampling_points)
        velocity_z = np.gradient(spline[2], sampling_points)
        velocity_yaw = np.gradient(spline[3], sampling_points)

        # get the absolute max velocity of the UAV
        max_velocity_x = max(abs(max(velocity_x)), abs(min(velocity_x)))
        max_velocity_y = max(abs(max(velocity_y)), abs(min(velocity_y)))
        max_velocity_z = max(abs(max(velocity_z)), abs(min(velocity_z)))
        max_velocity_yaw = max(abs(max(velocity_yaw)), abs(min(velocity_yaw)))

        # get the scaling factor for the velocity
        scaling_factor_x = max_velocity_x / waypoints[0].velocity_x
        scaling_factor_y = max_velocity_y / waypoints[0].velocity_y 
        scaling_factor_z = max_velocity_z / waypoints[0].velocity_z 
        scaling_factor_yaw = max_velocity_yaw / waypoints[0].velocity_yaw 
        self.get_logger().info("x scaling factor " + str(scaling_factor_x))
        self.get_logger().info("y scaling factor " + str(scaling_factor_y))
        self.get_logger().info("z scaling factor " + str(scaling_factor_z))
        self.get_logger().info("yaw scaling factor " + str(scaling_factor_yaw))
        scaling_factor = max(scaling_factor_x, scaling_factor_y, scaling_factor_z, scaling_factor_yaw)
        self.get_logger().info("scaling factor " + str(scaling_factor))

        # resampling the spline with the new velocity scaling factor
        sampling_points = np.linspace(0, 1, num=int(self.control_rate * scaling_factor))
        spline = interpolate.splev(sampling_points, tck, ext=3)

        # generate the path from spline
        path = [[spline[0][i], spline[1][i], spline[2][i], spline[3][i]] for i in range(len(spline[0]))]

        return path

    def publish_path_status(self, value):
        """
        Publishes a boolean value to the path status publisher

        Parameters:
        value (Boolean): The value to be published
        """
        msg = Bool()
        msg.data = value
        self.path_status_publisher.publish(msg)
    
    def publish_trajectory_command(self, action, x, y, z, yaw):
        """
        Publishes desired velocities to PX4 via the TrajectorySetpoint message

        Parameters:
        x (float): The desired x velocity/position of the drone
        y (float): The desired y velocity/position of the drone
        z (float): The desired z velocity/position of the drone
        yaw (float): The desired yaw velocity/angle of the drone
        """
        msg = TrajectorySetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        if (action == "velocity"): 
            msg.position = [float('nan'), float('nan'), float('nan')]
            msg.velocity = [x, y, z]
            msg.acceleration = [float('nan'), float('nan'), float('nan')]
            msg.yawspeed = yaw
        elif (action == "position"):
            msg.position = [x, y, z]
            msg.velocity = [float('nan'), float('nan'), float('nan')]
            msg.acceleration = [float('nan'), float('nan'), float('nan')]
            msg.yaw = yaw
        self.trajectory_publisher.publish(msg)
    
    def timer_callback(self):
        """
        Timer callback function
        """
        if self.uav_pose is None:
            return
        position = self.uav_pose.position
        orientation = self.uav_pose.q
        heading = Rotation.from_quat([orientation[3], orientation[0], orientation[1], orientation[2]]).as_euler('zyx')

        if self.mode == "takeoff":
            self.publish_offboard_mode("position")
            # if the UAV is not armed, arm it
            if not self.arm_status:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0) # Set to offboard mode
                self.arm()
            
            x = self.path[0][0]
            y = self.path[0][1]
            z = self.path[0][2]
            yaw = self.path[0][3]
            # self.get_logger().info("Desired: " + str(x) + " " + str(y) + " " + str(z))
            # self.get_logger().info("Current: " + str(position[0]) + " " + str(position[1]) + " " + str(position[2]))
            self.publish_trajectory_command("position", x, y, z, yaw)
            # self.publish_path_status(False)

            # check if the UAV has reached the desired position
            current_position = np.array([position[0], position[1], position[2]])
            desired_position = np.array([x, y, z])
            distance = np.linalg.norm(current_position - desired_position)
            # self.get_logger().info("distance error: " + str(distance))
            yaw_diff = abs(heading[0] - yaw) % 3.1415
            # self.get_logger().info("yaw error: " + str(yaw_diff))
            if distance < 1 and yaw_diff < 0.1:
                self.mode = "position"
                self.get_logger().info("Takeoff completed")
                self.publish_path_status(True)

        elif self.mode == "position":
            self.publish_offboard_mode("position")
            x = self.path[0][0]
            y = self.path[0][1]
            z = self.path[0][2]
            yaw = self.path[0][3]
            self.publish_trajectory_command("position", x, y, z, yaw)

        elif self.mode == "waypoints":
            self.publish_offboard_mode("velocity")
            if len(self.path) < 2:
                #self.get_logger().info("Path completed")
                # TODO: inform state machine that path is completed
                x_velocity_cmd = 0.
                y_velocity_cmd = 0.
                z_velocity_cmd = 0.
                yaw_velocity_cmd = 0.
                self.publish_trajectory_command("velocity", x_velocity_cmd, y_velocity_cmd, z_velocity_cmd, yaw_velocity_cmd)
                self.publish_path_status(True)
            
            else:
                # the first waypoint in the path 
                x = self.path[0][0]
                y = self.path[0][1]
                z = self.path[0][2]
                yaw = self.path[0][3]

                '''
                # calculate desired velocity in the path
                x_next = self.path[1][0]
                y_next = self.path[1][1]
                z_next = self.path[1][2]
                yaw_next = self.path[1][3]
                x_velocity = (x_next - x) / self.sampling_duration
                y_velocity = (y_next - y) / self.sampling_duration
                z_velocity = (z_next - z) / self.sampling_duration
                yaw_velocity = (yaw_next - yaw) / self.sampling_duration
                '''
                x_velocity = self.path_velocity[0][0]
                y_velocity = self.path_velocity[0][1]
                z_velocity = self.path_velocity[0][2]
                yaw_velocity = self.path_velocity[0][3]

                # pop the first waypoint from the path
                self.path.pop(0)
                self.path_velocity.pop(0)

                # calculate velocity commands from PID 
                x_velocity_cmd = self.pid_x.update(position[0], x, 1/self.control_rate) + x_velocity
                y_velocity_cmd = self.pid_y.update(position[1], y, 1/self.control_rate) + y_velocity
                z_velocity_cmd = self.pid_z.update(position[2] + 4, z, 1/self.control_rate) + z_velocity
                yaw_velocity_cmd = self.pid_yaw.update(heading[0], yaw, 1/self.control_rate) + yaw_velocity
                #self.get_logger().info('pose x: ' + str(position[0]) + " next x: " + str(x))
                #self.get_logger().info('pose y: ' + str(position[1]) + " next y: " + str(y))
                #self.get_logger().info('pose z: ' + str(position[2] + 4) + " next z: " + str(z))
                #self.get_logger().info("velocity published: " + str(x_velocity_cmd) + " " + str(y_velocity_cmd) + " " + str(z_velocity_cmd))

                # publish the velocity commands
                self.publish_trajectory_command("velocity", x_velocity_cmd, y_velocity_cmd, z_velocity_cmd, yaw_velocity_cmd)

                # publish path completion status
                # self.publish_path_status(True)
                

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
