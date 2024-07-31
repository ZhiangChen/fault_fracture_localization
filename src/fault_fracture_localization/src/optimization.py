import rclpy
from rclpy.node import Node
from custom_msgs.msg._trajectory_coordinates import TrajectoryCoordinates
from sensor_msgs.msg import Image
import numpy as np
from scipy.interpolate import interpolate


class Optimization(Node):
    def __init__(self):
        super().__init__("optimization")
        self.trajectory_publisher = self.create_publisher(TrajectoryCoordinates, "trajectory", 10)
        self.heatmap_subscriber = self.create_subscription(Image, "heatmap", self.heatmap_callback(), 10)
        # TODO subsribe to state

        #Number of points/derivatives per unit
        self.point = 10

    def heatmap_callback(self, data):
        pass


       
    def cmd_callback(self, data):
        # TODO replace x and y w/ actual datapoint
        x = np.array([1,2,3,4])
        y = np.array([3,4,5,6])
        t, c, k = interpolate.splrep(x, y, s = 0, k = 2)
        xmin, xmax = x.min, x.max
        xx = np.linspace(xmin, xmax, int(self.point * (x.max - x.min)))
        spline = interpolate.BSpline(t, c, k, extrapolate = False)
        #self.publish_trajectory_coordinates(spline(xx))


    def publish_trajectory_coordinates(self, spline):
        msg = TrajectoryCoordinates()
        msg.length = len(spline)
        msg.position = spline
        self.publisher.publish(msg)


def main(args = None):
    rclpy.init(args=args)
    optimization = Optimization()
    rclpy.spin(optimization)
    optimization.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()





