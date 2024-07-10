import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

model = YOLO('fracture-detector.pt') #Or whatever model we are trying to load

class Perception(Node):
    def __init__(self):
        super().__init__("perception")

        self.subscription = self.create_subscription(Image, "/camera", self.subscription_callback, 10)
        self.publisher = self.create_publisher(Image, "mask", 10)
        self.bridge = CvBridge()

    def subscription_callback(self, data):

        to_cv = self.bridge.imgmsg_to_cv2(data, desired_encoding = "bgr8")
        img = cv2.resize(to_cv, (640,640))
        results = model(img)[0]
        final = self.edge_mask(results)

        cv2.imshow("Detected Frame", results.plot())
        if (final is not None):
            cv2.imshow("Segmented", final)
        cv2.waitKey(1)
        self.publisher.publish(self.bridge.cv2_to_imgmsg(results.plot(), "bgr8"))

    def edge_mask(self, data):
        if (data.masks is None):
            return None
        contours = np.array(data.masks.xy, dtype = np.int32)
        edges = cv2.Canny(data.orig_img, 200, 250)
        mask = np.zeros(edges.shape)
        cv2.drawContours(mask, contours, -1, color=1, thickness = cv2.FILLED)
        edges[mask == 0] = 0
        return edges


def main(args=None):
    rclpy.init(args=args)
    perception = Perception()
    rclpy.spin(perception)
    perception.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

