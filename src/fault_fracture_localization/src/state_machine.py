import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from custom_msgs.srv._waypoint_service import WaypointService
from custom_msgs.msg._waypoint import Waypoint
from sensor_msgs.msg import Image
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from transitions import Machine


class StateMachine(Node):
    def __init__(self):

        timer_callback_group = MutuallyExclusiveCallbackGroup()
        heatmap_callback_group = MutuallyExclusiveCallbackGroup()
        super().__init__("statemachine")
        self.waypoint_client = self.create_client(WaypointService, "waypoint")
        while not self.waypoint_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waypoint service not availible, retrying")
        self.heatmap_subscriber = self.create_subscription(Image, "heatmap", self.heatmap_callback, 10, callback_group=heatmap_callback_group)
        self.timer_period = .05
        self.timer = self.create_timer(self.timer_period, self.timer_callback, callback_group=timer_callback_group)
        self.time = 0
        


    def waypoint_request(self, x, y , z, yaw, x_vel, y_vel, z_vel, yaw_vel):
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
        req.waypoint = waypoint
        self.future = self.waypoint_client.call_async(req)
        #rclpy.spin_until_future_complete(self, self.future)
        #return self.future.result()

    def heatmap_callback(self, data):
        # TODO take in heatmap and perform PCA
        #self.publish_waypoint(5.,5.,5.,0.)
        pass


    def timer_callback(self):
        self.waypoint_request(5.,5.,5.,5.,5.,5.,5.,5.)
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
