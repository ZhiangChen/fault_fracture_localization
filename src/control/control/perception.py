import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO

model = YOLO('fracture-dectector.pt') #Or whatever model we are trying to load

class Perception(Node):
    def __init__(self):
        super().__init__("perception")

        self.subscription = self.create_subscription(Image, "/camera", self.subscription_callback, 10)
        self.publisher = self.create_publisher(Image, "mask", 10)
        self.bridge = CvBridge()

    def subscription_callback(self, data):

        to_cv = self.bridge.imgmsg_to_cv2(data, desired_encoding = "bgr8")
        path = "/home/frank/fracture-mapping/images/" + str(self.number) + ".png"
        img = cv2.resize(to_cv, (640,640))
        results = model(img)[0].plot()
        cv2.imshow("Detected Frame", results)
        cv2.waitKey(1)
        self.publisher.publish(self.bridge.cv2_to_imgmsg(results, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    perception = Perception()
    rclpy.spin(perception)
    perception.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()



