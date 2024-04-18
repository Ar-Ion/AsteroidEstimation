import numpy as np
import pickle
import os
import quaternion
from datetime import datetime

from . import Frontend
from feature_descriptor.camera_utils import Extrinsics, Intrinsics

class DatasetFrontend(Frontend):

    def __init__(self, evaluator, motion_model, folder="/home/arion/AsteroidDataset"):
        
        self._evaluator = evaluator
        self._motion_model = motion_model

        self._train_folder = os.path.join(folder, "train")
        self._test_folder = os.path.join(folder, "train")

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
        while True:
            if self._id < self._size:
                filename = os.path.join(self._folder, str(self._id).zfill(6) + ".pickle")
                self._id += 1

                with open(filename, 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                if self._id % 100 == 0:
                    print("Generated " + str(self._id) + " synthetic payloads")

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
