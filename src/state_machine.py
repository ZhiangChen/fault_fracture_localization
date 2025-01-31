#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from rclpy.clock import Clock
from fault_fracture_localization.srv._waypoint_service import WaypointService
from fault_fracture_localization.msg._waypoint import Waypoint
from sensor_msgs.msg import Image
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from transitions import Machine
import numpy as np
from cv_bridge import CvBridge
import cv2
from collections import deque
import networkx as nx
import time
import threading


class StateMachine(Node):
    def __init__(self):
        super().__init__("state_machine")

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ("takeoff_height", rclpy.Parameter.Type.DOUBLE),
                ("waypoint_distance", rclpy.Parameter.Type.DOUBLE),
                ("desired_velocity", rclpy.Parameter.Type.DOUBLE),
            ]
        )

        # QOS profiles
        qos_best_effort_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Callback group declarations
        # Removed explicit callback groups to simplify threading <-- Changed

        # Subscriptions
        self.heatmap_subscriber = self.create_subscription(
            Image, "mask", self.mask_callback, 10
        )
        self.path_status_subscriber = self.create_subscription(
            Bool, "path_status", self.path_status_callback, 10
        )
        self.uav_pose_subscription = self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self.uav_pose_callback, qos_best_effort_profile
        )
        self.exploration_subscription = self.create_subscription(
            Image, "local_exploration", self.exploration_callback, 10
        )

        # Publishers
        self.state_publisher = self.create_publisher(String, "state", 10)

        # Clients
        self.waypoint_client = self.create_client(WaypointService, "waypoints")
        while not self.waypoint_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waypoint service not available, retrying")

        # Timer
        self.timer_period = 0.5  # seconds
        self.state_timer = self.create_timer(self.timer_period, self.state_timer_callback)  # <-- Changed

        # UAV data
        self.uav_pose = None  # Current UAV Pose
        self.initial_pose = None  # Where the UAV takes off and will return to
        self.branch_points = []  # Points where fault branches exist on the fault
        self.bridge = CvBridge()

        # Path Planning
        self.takeoff_height = -self.get_parameter("takeoff_height").value  # negate because of coordinate systems
        self.waypoint_distance = self.get_parameter("waypoint_distance").value
        self.desired_velocity = self.get_parameter("desired_velocity").value
        self.path_completion = False  # If the current path destination has been reached
        self.waypoint_queue = []  # Queue of waypoints since we publish in batches of 4
        self.previous_pose = None  # Previous pose UAV saw and recorded
        self.fault_detected = False  # Keeps if a fault was detected in the last mask message that was received
        self.path_status = False
        self.travel_dist = 4 # Number of units forward we use to plot the next point forward for waypoint calculations
        self.waypoint_throttle_duration = 3 # Time till next waypoint update
        self.last_waypoint_time = None # Last time a waypoint was published

        # Flags for state transitions
        self.prompt_needed = False

        # State machine
        states = [
            "idle",         # UAV is in pretakeoff, unarmed and is currently not doing anything
            "initiation",   # Initiation procedure where user inputs information is taking place, UAV is still unarmed
            "takeoff",      # UAV is arming and taking off, and reaching a predetermined hover distance
            "following",    # UAV is following a fault and updating the heatmap
            "searching",    # UAV is searching for a fault or fault continuation and is currently hovering in place
            "return",       # UAV is returning to the startpoint
            "resuming",     # UAV is recovering its state and in the process of resuming the search
            "hover",        # UAV is hovering in place
            "prompting",    # UAV is prompting user for feedback
        ]  # states for the FSM

        transitions = [
            {'trigger': 'startup', 'source': 'idle', 'dest': 'initiation'},
            {'trigger': 'takeoff_trigger', 'source': 'initiation', 'dest': 'takeoff'},
            {'trigger': 'resuming_trigger', 'source': 'takeoff', 'dest': 'resuming'},
            {'trigger': 'start_search', 'source': 'takeoff', 'dest': 'searching'},
            {'trigger': 'fault_detected_trigger', 'source': 'searching', 'dest': 'following'},
            {'trigger': 'end_trace_trigger', 'source': 'following', 'dest': 'searching'},
            {'trigger': 'none_found_trigger', 'source': 'searching', 'dest': 'prompting'},
            {'trigger': 'new_point_trigger', 'source': 'prompting', 'dest': 'resuming'},
            {'trigger': 'arrived_trigger', 'source': 'resuming', 'dest': 'searching'},
        ]

        self.machine = Machine(model=self, states=states, transitions=transitions, initial="idle",
                               auto_transitions=False, ignore_invalid_triggers=True)

        # Assign state entry callbacks 
        self.machine.on_enter_initiation('on_enter_initiation')
        self.machine.on_enter_takeoff('on_enter_takeoff')
        self.machine.on_enter_searching('on_enter_searching')
        self.machine.on_enter_following('on_enter_following')
        self.machine.on_enter_prompting('on_enter_prompting')
        self.machine.on_enter_resuming('on_enter_resuming')


    # State entry methods
    def on_enter_initiation(self):
        self.get_logger().info("Entering initiation state...")
        if self.uav_pose is None:
            self.get_logger().info("Waiting for UAV pose...")
            return  # Wait until pose is available
        self.initial_pose = self.uav_pose
        self.takeoff_trigger()

    def on_enter_takeoff(self):
        self.get_logger().info("Entering takeoff state...")
        self.last_waypoint_time = self.get_clock().now()
        if self.uav_pose is None:
            self.get_logger().warn("UAV pose is not available. Cannot take off.")
            return
        x = self.uav_pose.position[0]
        y = self.uav_pose.position[1]
        self.waypoint_request("takeoff", x, y, self.takeoff_height, 0., 0., 0., 0., 0.)
        # Transition to searching once takeoff is confirmed via path_status_callback

    def on_enter_searching(self):
        self.get_logger().info("Entering searching state...")
        # Initialize search parameters if needed
        # Example: Start processing exploration data or heatmaps

    def on_enter_following(self):
        self.get_logger().info("Entering following state...")
        # Initialize following parameters if needed

    def on_enter_prompting(self):
        self.get_logger().info("Entering prompting state...")
        # Prompt user for input or take necessary actions
        self.prompt_needed = True  # Example flag

    def on_enter_resuming(self):
        self.get_logger().info("Entering resuming state...")
        # Resume previous operations
        # Transition back to searching once resumed

    # Callback methods
    def state_timer_callback(self):
        self.publish_state(self.state)
        self.get_logger().info(f"Current state: {self.state}")  # Changed from .info to include state

        if self.state == "idle":
            self.startup()

        # Handle flags to trigger transitions
        if self.state == "takeoff" and self.check_path_status():
            self.start_search()

        if self.state == "searching":
            if self.fault_detected:
                self.fault_detected_trigger()
                self.fault_detected = False  
            elif self.prompt_needed:
                self.none_found_trigger()
                self.prompt_needed = False  

        if self.state == "following":
            # Implement logic to detect when to end trace
            # Example: based on some condition
            # if trace_ended:
            #     self.end_trace_trigger()
            pass

        if self.state == "prompting":
            # Implement user prompt handling
            # Example: after user input is received
            # self.new_point_trigger()
            pass

    def mask_callback(self, image):
        """
        Handles the masked fault images received from the perception node by performing a convolution
        and determining where the UAV should go next if it is in the correct mode.
        """
        self.get_logger().info("Mask received.")
        self.previous_pose = self.uav_pose
        to_cv = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        probability_map = self.conv(to_cv)
        max_index = probability_map.argmax()

        if np.amax(probability_map) == 0:
            self.fault_detected = False
            self.get_logger().info("No fault found!")
            return

        self.fault_detected = True
        indices = np.unravel_index(max_index, probability_map.shape)
        y, x, z = indices  
        x_diff = x - (probability_map.shape[1] // 2)
        omega = np.arctan2(y, x_diff)

        # Calculate new waypoint based on current position and omega
        '''
        if self.uav_pose is not None:
            position = self.uav_pose.position
            orientation = self.uav_pose.q
            heading = Rotation.from_quat([orientation[3], orientation[0], orientation[1], orientation[2]]).as_euler('zyx')
            new_x = position[0] + self.waypoint_distance * np.cos(omega)
            new_y = position[1] + self.waypoint_distance * np.sin(omega)
            self.get_logger().info(f"Moving to new waypoint: ({new_x}, {new_y}, {self.takeoff_height}) with heading {omega}")

            # Example waypoint request, will change later
            self.waypoint_request(
                action="waypoints",
                x=new_x,
                y=new_y,
                z=self.takeoff_height,
                yaw=omega,
                x_vel=self.desired_velocity * np.cos(omega),
                y_vel=self.desired_velocity * np.sin(omega),
                z_vel=0,
                yaw_vel=0
            )
        else:
            self.get_logger().warn("UAV pose is not available. Cannot compute new waypoint.")
        '''

    def path_status_callback(self, msg):
        """
        Callback function for the path_status subscription. Alerts the node when
        the current trajectory is completed
        """
        # self.get_logger().info("Received path status")
        self.path_status = msg.data

    def uav_pose_callback(self, msg):
        """
        Callback function for VehicleOdometry subscription. Updates current UAV pose
        """
        # self.get_logger().info("UAV pose received")
        self.uav_pose = msg

    def exploration_callback(self, msg):
        """
        Callback function for exploration subscription

        Parameters:
        msg (Image): A ROS image representing the exploration status
        """

        current_time = self.get_clock().now()
        time_since_last = current_time - self.last_waypoint_time
        if time_since_last <= rclpy.duration.Duration(seconds=self.waypoint_throttle_duration):
            return

        # Update the last waypoint time
        self.last_waypoint_time = current_time

        self.get_logger().info("Exploration map received.")
        to_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        height, width = to_cv.shape
        frontier = np.where(to_cv == 155)
        self.get_logger().info(f"Number of frontier points: {len(frontier[0])}")
        coordinates = np.stack((frontier[0], frontier[1]))
        current_point = (height // 2, width // 2)
        if len(frontier[0]) > 0:
            current_point = [current_point[0], current_point[1]]
            difference = coordinates - np.array(current_point).reshape(2, 1)
            dist = np.sum(difference**2, axis=0)
            index = np.argmin(dist)
            closest = [frontier[0][index], frontier[1][index]]
            omega = np.arctan2(closest[1] - current_point[1], closest[0] - current_point[0])
            self.get_logger().info(f"Angle to next point: {omega}")
            
            # If we are following, then we can query the next point
            if self.state == "following":
                # Calculate displacement based on omega
                x = self.uav_pose.position[0]
                y = self.uav_pose.position[1]
                z = self.uav_pose.position[2]
                orientation = self.uav_pose.q
                heading = Rotation.from_quat([orientation[3], orientation[0], orientation[1], orientation[2]]).as_euler('zyx')
                x_disp = self.travel_dist * np.cos(omega)
                y_disp = self.travel_dist * np.sin(omega)

                # Send waypoint request
                self.waypoint_request("position", x + x_disp, y + y_disp, z, heading, 5., 5., 0., 0.)
                self.get_logger().info("Waypoint request sent.")


        else:
            self.get_logger().info("No frontier point found :(")


    # Utility methods
    def waypoint_request(self, action, x, y, z, yaw, x_vel, y_vel, z_vel, yaw_vel):
        """
        Sends a waypoint for the UAV to travel to.
        """
        self.get_logger().info("Waypoint requested")
        waypoint = Waypoint()
        waypoint.timestamp = int(Clock().now().nanoseconds / 1000)
        waypoint.x = float(x)
        waypoint.y = float(y)
        waypoint.z = float(z)
        waypoint.yaw = float(yaw)
        waypoint.velocity_x = float(x_vel)
        waypoint.velocity_y = float(y_vel)
        waypoint.velocity_z = float(z_vel)
        waypoint.velocity_yaw = float(yaw_vel)

        req = WaypointService.Request()
        req.action = action
        req.waypoints = [waypoint]

        self.future = self.waypoint_client.call_async(req)
        self.future.add_done_callback(self.waypoint_response_callback) 

    def waypoint_response_callback(self, future):
        """
        Callback for handling the response from the waypoint service
        """
        try:
            response = future.result()
            self.get_logger().info("Waypoint service response received")
            # Handle response if needed
        except Exception as e:
            self.get_logger().error(f"Waypoint service call failed: {e}")

    def conv(self, image):
        """
        Applies convolution to the image
        """
        kernel = np.ones((7, 7), np.float32) / 49  # Normalized kernel
        return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    def publish_state(self, data):
        """
        Publishes the current state of the machine

        Parameters:
        data (String): The current state of the machine
        """
        msg = String()
        msg.data = data
        self.state_publisher.publish(msg)

    def check_path_status(self):
        """
        Checks if the path has been completed.

        Returns:
        Boolean: True if path has been completed, False otherwise
        """
        if self.path_status:
            self.path_status = False
            return True
        return False


def main(args=None):
    rclpy.init(args=args)
    state_machine = StateMachine()
    executor = MultiThreadedExecutor()
    executor.add_node(state_machine)
    try:
        executor.spin()
    finally:
        state_machine.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

