#!/usr/bin/env python3
import rclpy
import transforms3d
from scipy.spatial.transform import Rotation
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import Pose
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO
from collections import deque


model = YOLO('src/fault_fracture_localization/fracture-detector.engine', task='segment') 

class Perception(Node):
    def __init__(self):
        super().__init__("perception")

        self.declare_parameters(
            namespace = '',
            parameters = [
                ("fx", rclpy.Parameter.Type.DOUBLE),
                ("fy", rclpy.Parameter.Type.DOUBLE),
                ("cx", rclpy.Parameter.Type.DOUBLE),
                ("cy", rclpy.Parameter.Type.DOUBLE),
                ("dem_spacing", rclpy.Parameter.Type.INTEGER),
                ("sparse_dem_spacing", rclpy.Parameter.Type.INTEGER),
                ("exploration_window", rclpy.Parameter.Type.INTEGER),
                ]
        )

        # QOS profiles
        qos_best_effort_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Callback groups
        camera_callback_group = MutuallyExclusiveCallbackGroup()
        odometry_callback_group = MutuallyExclusiveCallbackGroup()

        # Subscriptions
        self.camera_subscription = self.create_subscription(Image, "/camera", self.camera_callback, 10, callback_group=camera_callback_group)
        self.uav_pose_subscription = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.uav_pose_callback, qos_best_effort_profile, callback_group=odometry_callback_group)
        self.state_subscription = self.create_subscription(String, "state", self.state_callback, 10, callback_group=odometry_callback_group)

        # Publishers
        self.segmentation_publisher = self.create_publisher(Image, "mask", 10)
        self.exploration_publisher = self.create_publisher(Image, "explored", 10)
        # self.heatmap_publisher = self.create_publisher(Image, "heatmap", 10)

        # Heatmap Generation
        self.uav_pose_cache = deque(maxlen=200)
        #self.camera_intrinsic = np.array([[861.923198, 0, 960.0],
        #              [0, 980.985917, 540.0],
        #              [0, 0, 1]])
        self.camera_intrinsic = np.array([[self.get_parameter("fx").value, 0, self.get_parameter("cx").value],
                                          [0, self.get_parameter("fy").value, self.get_parameter("cy").value],
                                        [0, 0, 1]])  # Hardcoded intrinsic matrix 
        self.dem_spacing = self.get_parameter("dem_spacing").value
        self.sparse_dem_spacing = self.get_parameter("sparse_dem_spacing").value
        self.bridge = CvBridge()
        self.dem  = self.generate_dem(1000, 500, self.dem_spacing) # TODO replace with actual dem 
        self.sparse_dem = self.generate_dem(1000, 500, self.spare_dem_spacing) # Heatmap used to bound the area needed for actual computation
        self.exploration_window = self.get_parameter("exploration_window").value # length and width of exploration map
        self.heatmap = self.generate_heatmap(1920, 1080) # Persistent heatmap of the area
        self.history = np.zeros_like(self.heatmap, dtype = np.uint32) # Holds how many times any given cell of the heatmap has been seen
        self.explored = np.zeroes_like(self.heatmap, dtype = np.uint32) # Marks whether a point has already been explored or not
        self.state = None # Current state the UAV is in



    def uav_pose_to_string(self, uav_pose):
        """
        Generates string of uav pose for debugging purposes

        Parameters:
        uav_pose (Pose): The pose of the uav to be turned into a string

        Returns:
        String: The pose as a string
        """

        first = "POSITION: " + str(uav_pose.position.x) + " " + str(uav_pose.position.y) + " " + str(uav_pose.position.z) + " \n"
        second = "ORIENTATION " + str(uav_pose.orientation.x) + " " + str(uav_pose.orientation.y) + " " + str(uav_pose.orientation.z) + " " + str(uav_pose.orientation.w) 
        return first + second

    def euler_to_rot_matrix(self, roll, pitch, yaw):
        """
        Convert Euler angles to a rotation matrix.

        Parameters:
        roll (float): Rotation around the x-axis (in radians).
        pitch (float): Rotation around the y-axis (in radians).
        yaw (float): Rotation around the z-axis (in radians).

        Returns:
        numpy.ndarray: The 3x3 rotation matrix.
        """
        # Compute individual rotation matrices
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        # Combine the rotation matrices
        R = np.matmul(R_z, np.matmul(R_y, R_x))
        return R

    def ned_to_enu(self, position):
        """
        Converts a transformation matrix in NED coordinates to ENU

        Parameters:
        position (numpy.ndarray): a 4x4 matrix in homogoneous coordinates

        Returns:
        numpy.ndarray: a 4x4 transformed matrix in ENU coordinates
        """
        T = np.array([[0, 1, 0, 0],
                     [1, 0, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]])
        return np.matmul(T, position)

    def state_callback(self, msg):
        """
        Callback function for UAV state subscription. 

        Parameters:
        msg (String): A String containing the current state of the UAV
        """
        self.state = msg

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
        

    def camera_callback(self, image):
        """
        Callback function for camera subscription. Segments the image and processes it 
        before sending to be used to update heatmap

        Parameters:
        image (Image): A Image message published by the camera
        """
        current_time = self.get_clock().now().nanoseconds


        #if camera initializes faster than pose we just pass until pose updates
        if len(self.uav_pose_cache) == 0:
            return
        # Binary search on the pose cache
        l = -1
        r = len(self.uav_pose_cache)
        while (r - l > 1):
            m = int((l + r) / 2)
            if (current_time < self.uav_pose_cache[m][1]):
                r = m
            else:
                l = m
        #self.get_logger().info(str(l))
        uav_pose = self.uav_pose_cache[l][0]
        

        to_cv = self.bridge.imgmsg_to_cv2(image, desired_encoding = "bgr8")
        img = cv2.resize(to_cv, (640,640))
        results = model.predict(img, device="cuda")[0]
        pixels = self.edge_mask(results)
        self.update_heatmap(pixels, uav_pose)
        self.segmentation_publisher.publish(self.bridge.cv2_to_imgmsg(pixels.astype(np.uint8), "mono8"))

        # Get local exploration map
        position = uav_pose.position
        x_actual = position.x / self.dem_spacing
        y_actual = position.y / self.dem_spacing
        part = self.explored[x_actual - self.exploration_window // 2 : x_actual + self.exploration_window // 2 ][y_actual - self.exploration_window // 2 : y_actual + self.exploration_window // 2 ]
        self.exploration_publisher.publish(part)


    def edge_mask(self, data):
        """
        Takes in a segmented image, performs edge detection, and only keeps the edges 
        within the mask

        Parameters:
        data (Results): An ultralytics results object from semantic segmentation

        Returns:
        numpy.ndarray: A grayscale image with the edge detections of the fault within the masks
        """
        edges = cv2.Canny(data.orig_img, 200, 250)
        mask = np.zeros(edges.shape)
        if (data.masks is not None):
            for i in data.masks.xy:
                contours = np.array([i], dtype = np.int32)
                cv2.drawContours(mask, contours, -1, color=1, thickness = cv2.FILLED)
        edges[mask == 0] = 0
        return edges

    def generate_dem(self, width, height, spacing):
        # Stopgap function for now
        x_range = np.arange(-width // 2, width // 2 + spacing, spacing)
        y_range = np.arange(-height // 2, height // 2 + spacing, spacing)

        x, y = np.meshgrid(x_range, y_range)

        x = x.flatten()
        y = y.flatten()
        z = np.zeros_like(x)

        return np.vstack((x, y, z)).T

    def generate_heatmap(self, width, height):
        '''
        Generate empty heatmap from given DEM

        Documentation TODO after figuring out how function will work
        '''
        return np.zeros((height, width), dtype = np.uint32)

    def generate_extrinsic_matrix(self, uav_pose):
        '''
        Generates camera extrinsic matrix

        Parameters:
        uav_pose (Pose): A Pose object to be used when generating the extrinsic matrix

        Returns:
        numpy.ndarray: A 4x4 camera extrinsic matrix
        '''
        orientation = uav_pose.orientation
        position = uav_pose.position
        rotation_matrix = Rotation.from_quat([orientation.w, orientation.x, orientation.y, orientation.z])
        # uav matrix originally in NED
        uav_matrix = np.eye(4)
        uav_matrix[:3, :3] = rotation_matrix.as_matrix()
        uav_matrix[:3, 3] = np.array([position.x, position.y, position.z])
        
        #transform to ENU
        uav_matrix = self.ned_to_enu(uav_matrix)


        # camera pose coordinates
        x= .16233
        y = -.001
        z = .238
        translation = transforms3d.affines.compose([x, y, z], np.eye(3), np.ones(3))

        roll = 0
        pitch = 0
        yaw = np.pi/2
        camera_mount_matrix = np.eye(4)
        camera_mount_matrix[:3, :3] = self.euler_to_rot_matrix(roll, pitch, yaw)
        mount_uav_matrix = np.eye(4)
        # first parameter governs if camera swivels up or down
        mount_uav_matrix[:3, :3] = self.euler_to_rot_matrix(0, 0, 0)
        camera_uav_matrix = np.matmul(camera_mount_matrix, mount_uav_matrix)
        extrinsic = np.matmul(uav_matrix, camera_uav_matrix)


        #self.get_logger().info("ROTATION MATRIX: " + np.array2string(rotation))
        #camera_matrix = np.matmul(translation, rotation)
        #self.get_logger().info("CAMERA MATRIX: " + np.array2string(camera_matrix))
        #coordinate_transform = np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]])
        #camera_matrix = np.matmul(coordinate_transform, camera_matrix)

        #extrinsic_matrix = np.matmul(uav_matrix, camera_matrix)
        #self.get_logger().info("EXTRINSIC MATRIX: " + np.array2string(extrinsic_matrix))
        return extrinsic

    def points_to_pixels(self, points, extrinsic_matrix, height, width):
        '''
        Generates a mapping from DEM points to pixel coordinates

        Parameters:
        extrinsic_matrix (numpy.ndarra): The extrinsic matrix of the camera
        height: the height of the image being projected to in pixels
        width: the width of the image being projected to in pixels
        '''

        # To homogoneous points
        points_homogoneous = np.hstack((points, np.ones((len(points), 1))))
        ex_inv = np.linalg.inv(extrinsic_matrix)
        points_transformed = np.matmul(points_homogoneous, ex_inv.T)[:,:3]

        # project points
        projected = np.matmul(points_transformed, self.camera_intrinsic.T)
        pixels = projected / projected[:, -1].reshape(-1, 1)
        pixels[:, 2] = projected[:, 2]
        #x_min, x_max = np.min(pixels[:, 0]).astype(np.int32), np.max(pixels[:, 0]).astype(np.int32)
        #y_min, y_max = np.min(pixels[:, 1]).astype(np.int32), np.max(pixels[:, 1]).astype(np.int32)
        #self.history[y_min:y_max][x_min:x_max] += 1

        # update all pixels that have been seen

        valid_points = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        
        valid_indices = np.where(valid_points)[0]

        pixel_point_association = np.zeros((height, width), dtype=int) - 1
        valid_pixels = pixels[valid_points].astype(int)

        pixel_point_association[valid_pixels[:, 1], valid_pixels[:, 0]] = valid_indices

        return pixel_point_association, valid_indices


    def camera_projection(self, mask, uav_pose):
        '''
        Projects camera image to heatmap

        Parameters:
        mask (numpy.ndarray): A masked grayscale edge detection image
        uav_pose (Pose): The pose of the UAV to be used for the update

        '''
        width, height, spacing = 1920, 1080, 1
        points = self.sparse_dem
        extrinsic_matrix = self.generate_extrinsic_matrix(uav_pose)
        pixel_point_association, valid_indices = self.points_to_pixels(points, extrinsic_matrix, height, width)

        filtered_points = points[valid_indices]
        if len(filtered_points) == 0:
            return

        x_min, x_max = np.min(filtered_points[:, 0]), np.max(filtered_points[:, 0])
        y_min, y_max = np.min(filtered_points[:, 1]), np.max(filtered_points[:, 1])

        points = self.dem

        points = points[(points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)]

        pixel_point_association, valid_indices = self.points_to_pixels(points, extrinsic_matrix, height, width)
        points[valid_indices]

        img = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        heatmap_points = points.copy()
        mask_indices = img > 0
        masked_points = pixel_point_association[mask_indices]
        masked_points = masked_points[masked_points >= 0]
        heatmap_points[masked_points, 2] = 255

# Convert heatmap points to heatmap for visualization
        height_grid, width_grid = int(height/spacing), int(width/spacing)
        heatmap = np.zeros((height_grid, width_grid), dtype = np.uint8)
        heatmap_indices = (heatmap_points[:, 0] + width_grid // 2).astype(int), (heatmap_points[:, 1] + height_grid // 2).astype(int)
        valid_heatmap = (heatmap_indices[0] >= 0) & (heatmap_indices[0] < width_grid) & (heatmap_indices[1] >= 0) & (heatmap_indices[1] < height_grid)
        heatmap[heatmap_indices[1][valid_heatmap], heatmap_indices[0][valid_heatmap]] = heatmap_points[valid_heatmap, 2]
        self.history[heatmap_indices[1][valid_heatmap], heatmap_indices[0][valid_heatmap]] += 1
        if self.state == "searching" or self.state == "following":
            self.explored[heatmap_indices[1][valid_heatmap], heatmap_indices[0][valid_heatmap]] += 1
            self.explored[heatmap_indices[1][valid_heatmap], heatmap_indices[0][valid_heatmap]] %= 256
# Flip the heatmap vertically
        heatmap = np.flipud(heatmap)
        return heatmap
    
    def update_heatmap(self, mask, uav_pose):
        heatmap = self.camera_projection(mask, uav_pose)
        if heatmap is None:
            return
        # kernel for dilation
        kernel = np.ones((5, 5), dtype = np.uint8)
        heatmap = cv2.dilate(heatmap, kernel, iterations=1)
        heatmap = cv2.blur(heatmap, (5, 5))
        self.heatmap = np.where(self.history > 0, (self.heatmap * (self.history) + heatmap), 0)
        self.heatmap = np.divide(self.heatmap, self.history, out = np.zeros_like(self.heatmap), where=self.history!=0, casting="unsafe")
        # self.heatmap_publisher.publish(self.bridge.cv2_to_imgmsg(self.heatmap.astype(np.uint8), encoding="mono8"))



def main(args=None):
    rclpy.init(args=args)
    perception = Perception()
    executor = MultiThreadedExecutor()
    executor.add_node(perception)
    executor.spin()
    perception.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

