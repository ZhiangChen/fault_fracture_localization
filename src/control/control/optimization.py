import rclpy
from rclpy.node import Node
from custom_msgs.msg._trajectory_coordinates import TrajectoryCoordinates
import numpy as np
from scipy.interpolate import interpolate
import matplotlib.pyplot as plt


class Optimization(Node):
    def __init__(self):
        super().__init__("optimization")
        self.publisher = self.create_publisher(TrajectoryCoordinates, "trajectory", 10)
        # TODO subsribe to state

        #Number of points/derivatives per unit
        self.point = 10

       
    def cmd_callback(self, data):
        # TODO replace x and y w/ actual datapoint
        x = np.array([1,2,3,4])
        y = np.array([3,4,5,6])
        t, c, k = interpolate.splrep(x, y, s = 0, k = 2)
        xmin, xmax = x.min, x.max
        xx = np.linspace(xmin, xmax, int(self.point * (x.max - x.min)))
        spline = interpolate.BSpline(t, c, k, extrapolate = False)
        self.publish_trajectory_coordinates(spline(xx))


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





