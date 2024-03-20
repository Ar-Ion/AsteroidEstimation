import rclpy
import threading
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from . import Frontend

class ROSFrontend(Node, Frontend):

    def __init__(self, evaluator):
        super().__init__('feature_extractor_frontend')

        self._evaluator = evaluator
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

    def rgb_callback(self, msg):
        with self._lock:
            self._last_rgb_msg = msg
            self.on_potential_sync()
            

    def depth_callback(self, msg):
        with self._lock:
            self._last_depth_msg = msg
            self.on_potential_sync()

    def on_potential_sync(self):
        if self._last_rgb_msg.header.stamp == self._last_depth_msg.header.stamp:
            self.on_sync(self._last_rgb_msg.header.stamp.sec * 1e9 + self._last_rgb_msg.header.stamp.nanosec)

    # Timestamp in nanoseconds
    def on_sync(self, timestamp):
        image = self._cv_bridge.imgmsg_to_cv2(self._last_rgb_msg, "mono8")
        depth = self._cv_bridge.imgmsg_to_cv2(self._last_depth_msg, "32FC1")

        self._evaluator.on_input(timestamp, image, depth)

    def loop(self):
        rclpy.spin(self)

    def cleanup(self):
        self.destroy_node()
