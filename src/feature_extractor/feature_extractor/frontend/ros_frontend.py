import rclpy
import threading
import cv2
import numpy as np
import quaternion
import time

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
        self._last_rgb_msg = None
        self._last_depth_msg = None
        self._last_info_msg = None
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
        
        self.timer = self.create_timer(0.1, self.timer_callback)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        
        self._simulation_buffer = []
        self._simulation_buffer_size = 10000
        self.get_logger().info("ROS2 frontend initialized")

    def rgb_callback(self, msg):
        with self._lock:
            self._last_rgb_msg = msg

    def depth_callback(self, msg):
        with self._lock:
            self._last_depth_msg = msg

    def info_callback(self, msg):
        with self._lock:
            self._last_info_msg = msg

    def timer_callback(self):
        with self._lock:
            if self._last_rgb_msg != None and self._last_depth_msg != None and self._last_info_msg != None:
                rgb_stamp = self._last_rgb_msg.header.stamp
                depth_stamp = self._last_depth_msg.header.stamp        
            
                self.on_sync(rgb_stamp.sec * 1e9 + rgb_stamp.nanosec)

    # Timestamp in nanoseconds
    def on_sync(self, timestamp):
        intrinsics = Intrinsics(np.array(self._last_info_msg.k).reshape(3, 3))        
        image = self._cv_bridge.imgmsg_to_cv2(self._last_rgb_msg, "mono8")
        depth = self._cv_bridge.imgmsg_to_cv2(self._last_depth_msg, "32FC1")

        try:
            t = self._tf_buffer.lookup_transform("camera", "asteroid", rclpy.time.Time())
            
            trans = t.transform.translation
            rot = t.transform.rotation

            disp = np.array([trans.x, trans.y, trans.z])
            quat = np.quaternion(rot.w, rot.x, rot.y, rot.z)

            extrinsics = Extrinsics(disp, quat)
              
            if len(self._simulation_buffer) < self._simulation_buffer_size:
                self._simulation_buffer.append((timestamp, intrinsics, extrinsics, image, depth))
                self.get_logger().info("Acquired frame " + str(len(self._simulation_buffer)))
            
        except TransformException:
            self.get_logger().info("Still waiting for the camera to asteroid transform...")
            return

        if len(self._simulation_buffer) == self._simulation_buffer_size:
            for i, (timestamp, intrinsics, extrinsics, image, depth) in enumerate(self._simulation_buffer):
                self._motion_model.set_camera_intrinsics(intrinsics)
                self._motion_model.set_camera_extrinsics(extrinsics)
                self._evaluator.on_input(timestamp, image, depth)

                self.get_logger().info("Evaluated frame " + str(i))

            self._evaluator.on_finish()
            self._simulation_buffer.clear()


    def loop(self):
        self.get_logger().info("Running ROS2 frontend...")
        rclpy.spin(self)

    def cleanup(self):
        self.destroy_node()
