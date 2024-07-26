import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from custom_msgs.msg._waypoint import Waypoint
from sensor_msgs.msg import Image


class StateMachine(Node):
    def __init__(self):
        super().__init__("statemachine")
        self.waypoint_publisher = self.create_publisher(Waypoint, "waypoint", 10)
        self.heatmap_subscriber = self.create_subscription(Image, "heatmap", self.heatmap_callback, 10)


    def publish_waypoint(self, x, y , z, yaw):
        msg = Waypoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.x = x
        msg.y = y
        msg.z = z
        msg.yaw = yaw
        self.waypoint_publisher.publish(msg)

    def heatmap_callback(self, data):
        # TODO take in heatmap and perform PCA
        #self.publish_waypoint(5.,5.,5.,0.)
        pass

def main(args = None):
    rclpy.init(args=args)
    state_machine = StateMachine()
    rclpy.spin(state_machine)
    state_machine.destroy_node()
    rclpy.shutdown()



if __name__ == "__main__":
    main()
