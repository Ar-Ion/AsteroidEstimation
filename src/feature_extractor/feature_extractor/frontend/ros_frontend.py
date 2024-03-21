import rclpy
import threading
import cv2
import numpy as np
import quaternion

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from . import Frontend
from feature_extractor.camera_utils import Extrinsics, Intrinsics

class ROSFrontend(Node, Frontend):

    def __init__(self, evaluator, motion_model):
        super().__init__('feature_extractor_frontend')

        self._evaluator = evaluator
        self._motion_model = motion_model
        self._last_rgb_msg = Image()
        self._last_depth_msg = Image()
        self._lock = threading.Lock()
        self._cv_bridge = CvBridge()
                
        self._rgb_sub = self.create_subscription(
            Image,
            '/camera/rgb',
            self.rgb_callback,
            10
        )

        self._depth_sub = self.create_subscription(
            Image,
            '/camera/depth',
            self.depth_callback,
            10
        )

        self._cam_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/info',
            self.info_callback,
            10
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.get_logger().info("ROS2 frontend initialized")

    def rgb_callback(self, msg):
        with self._lock:
            self._last_rgb_msg = msg
            self.on_potential_sync()

    def depth_callback(self, msg):
        with self._lock:
            self._last_depth_msg = msg
            self.on_potential_sync()

    def info_callback(self, msg):
        intrinsics = Intrinsics(np.array(msg.k).reshape(3, 3))
        self._motion_model.set_camera_intrinsics(intrinsics)

    def on_potential_sync(self):
        if self._last_rgb_msg.header.stamp == self._last_depth_msg.header.stamp:
            self.on_sync(self._last_rgb_msg.header.stamp.sec * 1e9 + self._last_rgb_msg.header.stamp.nanosec)

    # Timestamp in nanoseconds
    def on_sync(self, timestamp):
        image = self._cv_bridge.imgmsg_to_cv2(self._last_rgb_msg, "mono8")
        depth = self._cv_bridge.imgmsg_to_cv2(self._last_depth_msg, "32FC1")

        try:
            t = self._tf_buffer.lookup_transform("camera", "asteroid", rclpy.time.Time())
            
            trans = t.transform.translation
            rot = t.transform.rotation

            disp = np.array([trans.x, trans.y, trans.z])
            quat = np.quaternion(rot.w, rot.x, rot.y, rot.z)

            extrinsics = Extrinsics(disp, quat)

            self._motion_model.set_camera_extrinsics(extrinsics)
            
        except TransformException:
            self.get_logger().info("Still waiting for the camera to asteroid transform...")
            return

        self._evaluator.on_input(timestamp, image, depth)

    def loop(self):
        self.get_logger().info("Running ROS2 frontend...")
        rclpy.spin(self)

    def cleanup(self):
        self.destroy_node()
