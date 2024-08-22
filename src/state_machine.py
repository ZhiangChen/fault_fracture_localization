#!/usr/bin/env python3
import rclpy
import time
import threading
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from rclpy.clock import Clock
from fault_fracture_localization.srv._waypoint_service import WaypointService
from fault_fracture_localization.msg._waypoint import Waypoint
from sensor_msgs.msg import Image
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from transitions import Machine
import numpy as np
from collections import deque
import networkx as nx
from std_msgs.msg import Bool
import cv2
from cv_bridge import CvBridge


class StateMachine(Node):
    def __init__(self):
        super().__init__("state_machine")

        self.declare_parameters(
            namespace = '',
            parameters = [
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
        machine_callback_group = MutuallyExclusiveCallbackGroup()
        timer_callback_group = MutuallyExclusiveCallbackGroup()
        mask_callback_group = MutuallyExclusiveCallbackGroup()
        odometry_callback_group = MutuallyExclusiveCallbackGroup()

        # Subscriptions
        self.heatmap_subscriber = self.create_subscription(Image, "mask", self.mask_callback, 10, callback_group=mask_callback_group)
        self.path_status_subscriber = self.create_subscription(Bool, "path_status", self.path_status_callback, 10, callback_group=odometry_callback_group)
        self.uav_pose_subscription = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.uav_pose_callback, qos_best_effort_profile, callback_group=odometry_callback_group)
        self.exploration_subscription = self.create_subscription(Image, "exploration", self.exploration_callback, 10, callback_group=odometry_callback_group)

        # Publishers
        self.state_publisher = self.create_publisher(String, "state", 10)
        # Clients
        self.waypoint_client = self.create_client(WaypointService, "waypoints")
        while not self.waypoint_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waypoint service not availible, retrying")

            
        self.time = 0
        self.timer_period = .5
        # self.machine_timer = self.create_timer(self.timer_period, self.machine_timer_callback, callback_group=machine_callback_group)
        self.state_timer = self.create_timer(self.timer_period, self.state_timer_callback, callback_group=timer_callback_group)

        # UAV data 
        self.uav_pose = None # Current UAV Pose
        self.initial_pose = None # Where the UAV takes off and will return to
        self.branch_points = [] # Points where fault branches exist on the fault
        self.bridge = CvBridge()

        # Path Planning 
        self.takeoff_height = -self.get_parameter("takeoff_height").value # negate because of coordinate systems
        self.waypoint_distance = self.get_parameter("waypoint_distance").value
        self.desired_velocity = self.get_parameter("desired_velocity").value
        self.path_completion = False # If the current path destination has been reached
        self.waypoint_queue = [] # Queue of waypoints since we publish in batches of 4
        self.previous_pose = None # Previous pose UAV saw and recorded
        self.fault_detected = False # Keeps if a fault was detected in the last mask message that was recieved
        self.path_status = False

        # State machine
        states = ["idle", # UAV is in pretakeoff, unarmed and is currently not doing anything
                  "initiation", # Initiation procedure where user inputs information is taking place, UAV is still unarmed
                  "takeoff", # UAV is arming and taking off, and reaching a predetermined hover distance
                  "following", # UAV is following a fault and updating the heatmap
                  "searching", # UAV is searching for a fault or fault continuation and is currently hovering in place
                  "return", # UAV is returning to the startpoint
                  "resuming", # UAV is recovering its state and in the process of resuming the search
                  "hover", # UAV is hovering in place
                  "prompting", # UAV is prompting user for feedback
                  ] # states for the FSM
        transitions = [
            { 'trigger': 'startup', 'source': 'idle', 'dest': 'initiation' }, # UAV starts up from idle
            { 'trigger': 'takeoff', 'source': 'initiation', 'dest': 'takeoff' }, # UAV arms and takes off from ground
            { 'trigger': 'resuming', 'source': 'takeoff', 'dest': 'resuming' }, # UAV finishes takeoff and resumes previous search
            { 'trigger': 'start_search', 'source': 'takeoff', 'dest': 'searching' }, # UAV starts a new search
            { 'trigger': 'fault_detected', 'source': 'searching', 'dest': 'following' }, # UAV finds a fault and starts to follow it
            { 'trigger': 'end_trace', 'source': 'following', 'dest': 'searching' }, # trace presumably ends, UAV goes back to searching
            { 'trigger': 'none_found', 'source': 'searching', 'dest': 'prompting' }, # no fault was found so the UAV prompts the user
            { 'trigger': 'new_point', 'source': 'prompting', 'dest': 'resuming' }, # UAV moves to location of new prompted point
            { 'trigger': 'arrived', 'source': 'resuming', 'dest': 'searching' }, # UAV arrives at the resuming point
            { 'trigger': 'arrived', 'source': 'resuming', 'dest': 'searching' }, # UAV arrives at the resuming point
            #{ 'trigger': 'hover', 'source': ['takeoff', 'following', 'searching', 'return', 'resuming'], 'dest': 'hover' },
            #{ 'trigger': 'return', 'source': ['takeoff', 'following', 'searching', 'hover', 'resuming'], 'dest': 'return' },
        ]
        self.machine = Machine(self, states=states, transitions=transitions, initial="idle")
        self.machine_interval = .1 # amount the machine thread will sleep after each loop

        machine_thread = threading.Thread(target=self.startup)
        machine_thread.start()
        

    def on_enter_takeoff(self):
        """
        This function will be automatically invoked when the state machine transitions to the takeoff
        state. The UAV attempts to take off after recieving the UAV pose and then transitions into the 
        search state
        """
        self.get_logger().info("taking off...")
        x = self.uav_pose.position[0]
        y = self.uav_pose.position[1]
        self.waypoint_request("takeoff", x, y, self.takeoff_height, 1.57, 0., 0., 0., 0.)
        while not self.check_path_status():
            rclpy.spin_once(self, timeout_sec=self.machine_interval)
        self.start_search()


    def on_enter_searching(self):
        """
        This function will be automatically invoked when the state machine transitions to the searching
        state. The UAV will attempt to detect a fault to a reasonable level. If it cannot detect the fault,
        it will prompt the user on what to do next
        """
        # Swerve camera
        seen = 0 # Amount of times the fault has been accurately detected in a short span on time
        misses = 0 # Number of times the fault hasnt been seen in a row
        while (seen < 5):
            if (self.fault_detected):
                seen += 1
                misses = 0
            else:
                misses += 1
                if (misses > 3):
                    # Prompt user for next action
                    pass
            rclpy.spin_once(self, timeout_sec=self.machine_interval)
        


    
    def on_enter_initiation(self):
        """
        Transition from idle state to the initiation state. This state will check the parameter
        inputs from the YAML file and then transition to the takeoff state
        """
        # TODO check YAML file

        self.get_logger().info("initiating...")
        # First save the point where UAV started 
        while(self.uav_pose is None):
            self.get_logger().info("waiting for pose...")
            rclpy.spin_once(self, timeout_sec=self.machine_interval)
        self.initial_pose = self.uav_pose
        
        self.takeoff()

    def on_enter_following(self):
        """
        This function will be automatically be invoked when the UAV enters the following state. The UAV
        will follow the fault until there is a period where the UAV is not sure that there is a fault, where it 
        will return to the search state
        """
        seen = 0
        misses = 0
        while (True): 
            if (self.fault_detected):
                seen += 1
                misses = 0
            else:
                misses += 1
                if (misses > 5):
                    self.end_trace()
            rclpy.spin_once(self, timeout_sec=self.machine_interval)

        # start following procedure

    def on_enter_resuming(self):
        # resume from last point
        pass

    def on_enter_prompting(self):
          # Prompt user for waypoint, etc
        pass

    def exploration_callback(self, msg):
        """
        Callback function for exploration subscription

        Parameters:
        msg (np.ndarray): A 2 dimensional array representing the exploration status of the DEM
        """
        pass

    def path_status_callback(self, msg):
        """
        Callback function for the path_status subscription. Alerts the node when
        the current trajectory is completed

        Parameters:
        msg (Bool): A boolean indicating if the path the UAV is currently following has been completed
        """
        # self.get_logger().info("recieved data")
        self.path_status = msg.data


    def uav_pose_callback(self, msg):
        """
        Callback function for VehicleOdometry subscription. Adds current pose and timestamp
        to cache.

        Parameters:
        msg (VehicleOdometry): A VehicleOdometry message published by PX4

        """
        # self.get_logger().info("pose recieved")
        self.uav_pose = msg

    def waypoint_request(self, action, x, y, z, yaw, x_vel, y_vel, z_vel, yaw_vel):
        """
        Sends a waypoint for the UAV to travel to. Waypoint requests will be held on to until the batch number has been
        reached, in which a request will be sent to the WaypointService client

        Parameters:
        action (String): The action that should be taken at the waypoint, such as hover, takeoff, etc
        x (float): the x coordinate that the UAV should take on at this waypoint
        y (float): the y coordinate that the UAV should take on at this waypoint
        z (float): the z coordinate that the UAV should take on at this waypoint
        yaw (float): the yaw coordinate that the UAV should take on at this waypoint
        x_vel (float): the velocity in the x direction the UAV should be travelling at when it reaches this waypoint
        y_vel (float): the velocity in the y direction the UAV should be travelling at when it reaches this waypoint
        z_vel (float): the velocity in the z direction the UAV should be travelling at when it reaches this waypoint
        yaw_vel (float): the velocity in the yaw direction the UAV should be travelling at when it reaches this waypoint

        """
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
        if action == "takeoff":
            req = WaypointService.Request()
            req.action = action
            req.waypoints = [waypoint]
            self.future = self.waypoint_client.call_async(req)

        else:
            self.waypoint_queue.append(waypoint)
            if len(self.waypoint_queue) == 2:
                req = WaypointService.Request()
                req.action = action
                req.waypoints = self.waypoint_queue
                self.future = self.waypoint_client.call_async(req)
                self.waypoint_queue.clear()
        #rclpy.spin_until_future_complete(self, self.future)
        #return self.future.result()

    def mask_callback(self, image):
        """
        Handles the masked fault images recieved from the perception node by performing a convolution
        and determining where the UAV should go next if it is in the correct mode. 

        Parameters:
        image: a ROS image which contains the masked fault
        """
        # TODO take in heatmap and perform PCA
        #self.publish_waypoint(5.,5.,5.,0.)
        #self.timer += 1
        #if self.timer == 20:
        #    self.Graph.add_node(self.uav_pose)
        #    if self.previous_pose is not None:
        #        self.Graph.add_edge(self.previous_pose, self.uav_pose)
        #    self.timer = 0

        self.previous_pose = self.uav_pose
        to_cv = self.bridge.imgmsg_to_cv2(image, desired_encoding = "bgr8")
        probability_map = self.conv(to_cv)
        max_index = probability_map.argmax()
        if np.amax(probability_map) == 0:
            self.fault_detected = False
            self.get_logger().info("No fault found!")
            return
        self.fault_detected = True
        indices = np.unravel_index(max_index, probability_map.shape)
        y, x, z = indices
        x_diff = x - (len(probability_map[0]) // 2)
        omega = np.arctan2(y, x_diff)

        # Find direction and amount to move in the next waypoint
        position = self.uav_pose.position
        orientation = self.uav_pose.q
        velocity = self.uav_pose.velocity
        angular_velocity = self.uav_pose.angular_velocity
        heading = Rotation.from_quat([orientation[3], orientation[0], orientation[1], orientation[2]]).as_euler('zyx')
        new_x = position[0] + self.waypoint_distance * np.cos(omega)
        new_y = position[1] + self.waypoint_distance * np.sin(omega)
        # self.get_logger().info(str(new_x) + " " + str(new_y))
        self.get_logger().info("height " + str(position[2]))

        if self.state == "searching":
            self.waypoint_request("waypoints", position[0], position[1], self.takeoff_height, heading[0], velocity[0], velocity[1], 0, angular_velocity[0])
            self.waypoint_request("waypoints", new_x, new_y, self.takeoff_height, omega, self.desired_velocity * np.cos(omega), self.desired_velocity * np.sin(omega), 0, 0)

    def conv(self, image):
        kernel = np.ones((7, 7))
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
        #self.get_logger().info(self.state)

    def check_path_status(self):
        """
        Checks if the path has been completed. 

        Returns:
        Boolean: True if path has been completed, False otherwise
        """
        if not self.path_status:
            return False
        else:
            self.path_status = False
            return True

    def state_timer_callback(self):
        self.publish_state(self.state)
        #self.get_logger().info(self.state)

    def machine_timer_callback(self):
        self.startup()

def main(args = None):
    rclpy.init(args=args)
    state_machine = StateMachine()
    executor = MultiThreadedExecutor()
    executor.add_node(state_machine)
    executor.spin()
    state_machine.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
