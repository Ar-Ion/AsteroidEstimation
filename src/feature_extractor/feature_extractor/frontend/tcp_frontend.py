import numpy as np
import socket
import pickle
import quaternion
from datetime import datetime

from . import Frontend
from feature_extractor.camera_utils import Extrinsics, Intrinsics

class TCPFrontend(Frontend):

    def __init__(self, evaluator, motion_model, host="127.0.0.1", port=42666):
        
        self._evaluator = evaluator
        self._motion_model = motion_model

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self._socket.connect((host, port))
            print("TCP frontend initialized")
        except:
            print("Failed to initialize TCP frontend")
            self._socket.close()
            self._socket = None

    def payload_callback(self, msg):
        (env_data, robot_data) = msg
        
        cam_data = robot_data[0]
        (cam_pose, k, image, depth) = cam_data
        (cam_trans, cam_rot) = cam_pose
        cam_quat = np.quaternion(*cam_rot)

        env_pose = env_data
        (env_trans, env_rot) = env_pose
        env_quat = np.quaternion(*env_rot)

        env2cam_trans = quaternion.rotate_vectors(cam_quat, env_trans - cam_trans)
        env2cam_quat = env_quat

        intrinsics = Intrinsics(k)
        extrinsics = Extrinsics(env2cam_trans, env2cam_quat)

        self._motion_model.set_camera_intrinsics(intrinsics)
        self._motion_model.set_camera_extrinsics(extrinsics)

        self._evaluator.on_input(image, depth)

    def loop(self):
        with self._socket as s:
            while True:
                frame_start = s.recv(1)

                if frame_start == b'\x7f':
                    payload = self.receive_payload(s)
                    msg = pickle.loads(payload)
                    self.payload_callback(msg)

    def receive_payload(self, s):
        packet_length = int.from_bytes(s.recv(4), byteorder='big')
        recv_length = 0
        payload = b''

        while recv_length < packet_length:
            remaining_length = packet_length - recv_length
            chunk_length = min(remaining_length, 4096)

            chunk = s.recv(chunk_length)
            payload += chunk

            recv_length += len(chunk)

        return payload

    def cleanup(self):
        self._socket.close()
