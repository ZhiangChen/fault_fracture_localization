#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from fault_fracture_localization.srv._waypoint_service import WaypointService
from fault_fracture_localization.msg._waypoint import Waypoint
from sensor_msgs.msg import Image
from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import Pose
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from transitions import Machine
from collections import deque
from scipy.spatial.transform import Rotation


class StateMachine(Node):
    def __init__(self):
        super().__init__("state_machine")

        # QOS profiles
        qos_best_effort_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Callback group declarations
        timer_callback_group = MutuallyExclusiveCallbackGroup()
        heatmap_callback_group = MutuallyExclusiveCallbackGroup()
        odometry_callback_group = MutuallyExclusiveCallbackGroup()

        # Subscriptions
        self.heatmap_subscriber = self.create_subscription(Image, "heatmap", self.heatmap_callback, 10, callback_group=heatmap_callback_group)
        self.uav_pose_subscription = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.uav_pose_callback, qos_best_effort_profile, callback_group=odometry_callback_group)
        # Clients
        self.waypoint_client = self.create_client(WaypointService, "waypoint")
        while not self.waypoint_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waypoint service not availible, retrying")

        # timer callback for main loop
        self.timer_period = .05
        self.timer = self.create_timer(self.timer_period, self.timer_callback, callback_group=timer_callback_group)
        self.time = 0

        # UAV data 
        self.uav_pose_cache = deque(maxlen=200)
        self.initial_pose = None # Where the UAV takes off and will return to
        self.branch_points = [] # Points where fault branches exist on the fault

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
        self.machine = Machine(model=self, states=states, transitions=transitions, initial="idle")
        

    def on_enter_takeoff(self):
        self.waypoint_request("takeoff", 0., 0., -100., 0., 0., 0.)
        self.waypoint_request("follow", 0., 0., -100., 0., 0., 0.)

    
    def on_enter_startup(self):
        #prompt for info

        # First save the point where UAV started 
        self.initial_pose = self.uav_pose_cache[-1][0]
        
        pass

    def on_enter_following(self):
        #start following procedure
        pass

    def on_enter_return(self):
        #return to startpoint
        pass

    def on_enter_searching(self):
        #search the area
        pass

    def on_enter_resuming(self):
        # resume from last point
        pass

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

    def waypoint_request(self, action, x, y , z, yaw, x_vel, y_vel, z_vel, yaw_vel):
        waypoint = Waypoint()
        waypoint.timestamp = int(Clock().now().nanoseconds / 1000)
        waypoint.x = x
        waypoint.y = y
        waypoint.z = z
        waypoint.yaw = yaw
        waypoint.velocity_x = x_vel
        waypoint.velocity_y = y_vel
        waypoint.velocity_z = z_vel
        waypoint.velocity_yaw = yaw_vel
        req = WaypointService.Request()
        req.action = action
        req.waypoint = waypoint
        self.future = self.waypoint_client.call_async(req)
        #rclpy.spin_until_future_complete(self, self.future)
        #return self.future.result()

    def heatmap_callback(self, data):
        # TODO take in heatmap and perform PCA
        #self.publish_waypoint(5.,5.,5.,0.)
        pass


    def timer_callback(self):
        if (self.time == 100):
            self.waypoint_request("takeoff", 0., 0., -50., 0., 0.,0.,0.,0.)
            self.waypoint_request("tracking", 0., 0., -50., 0., 5.,5.,5.,0.)
            self.waypoint_request("tracking", 50., 50., -20., 0., 5.,5.,5.,0.)
            self.waypoint_request("tracking", 50., 0., -40., 0., 5.,5.,5.,0.)
            self.waypoint_request("tracking", 25., 25., -20., 0., 5.,5.,5.,0.)
            self.waypoint_request("tracking", 0., 0., -20., 0., 5.,5.,5.,0.)

        self.time += 1


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
