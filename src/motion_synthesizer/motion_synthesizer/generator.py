import numpy as np
import quaternion
import torch

from astronet_frontends import DriveClientFrontend
from camera_utils import Extrinsics, Intrinsics, CameraProjection
from astronet_msgs import MotionData

class MotionGenerator:
    def __init__(self, client, server, input_size, output_size):
        self._client = client
        self._server = server
        self._input_size = input_size
        self._output_size = output_size
        self._count = 0

    def get_projection(self, data):
        env_pose = data.env_data.pose
        cam_pose = data.robot_data.cam_data[0].pose
        
        env_quat = np.quaternion(*env_pose.rot)
        cam_quat = np.quaternion(*cam_pose.rot)

        rot = np.quaternion(np.sqrt(2)/2, -np.sqrt(2)/2, 0, 0) # I found this matrix empirically... Ideally, use cam_quat instead

        env2cam_trans = quaternion.rotate_vectors(cam_quat, env_pose.trans - cam_pose.trans)
        env2cam_quat = rot * env_quat

        intrinsics = Intrinsics(data.robot_data.cam_data[0].k)
        extrinsics = Extrinsics(env2cam_trans, env2cam_quat)

        return CameraProjection(intrinsics, extrinsics)

    def loop(self):
        while self._count < self._output_size:
            if self._count != 0 and self._count*2 % self._input_size == 0:
                # We have iterated through the whole input dataset. The data must be reshuffled.
                self._client.send_event(DriveClientFrontend.Events.RESET)
        
            data1 = self._client.receive(blocking=True)
            data2 = self._client.receive(blocking=True)
            
            proj1 = self.get_projection(data1)
            proj2 = self.get_projection(data2)
            
            distance1 = torch.Tensor(data1.robot_data.cam_data[0].pose.trans - data1.env_data.pose.trans)
            distance2 = torch.Tensor(data2.robot_data.cam_data[0].pose.trans - data2.env_data.pose.trans)
                        
            data1_coords = torch.Tensor(data1.robot_data.cam_data[0].coords)
            data1_features = torch.Tensor(data1.robot_data.cam_data[0].features)
            data2_coords = torch.Tensor(data2.robot_data.cam_data[0].coords)
            data2_features = torch.Tensor(data2.robot_data.cam_data[0].features)
            
            env1_coords = proj1.camera2object(data1_coords)
            reproj1_coords = proj2.object2camera(env1_coords)

            non_nan_filter = ~torch.any(reproj1_coords.isnan(), dim=1)
            visible_filter = reproj1_coords[:, 2] < torch.norm(distance2)
            
            expected_kps = reproj1_coords[non_nan_filter & visible_filter, 0:2]
            expected_features = data1_features[non_nan_filter & visible_filter]
            actual_kps = data2_coords[:, 0:2]
            actual_features = data2_features
                        
            # plt.figure()
            # plt.scatter(actual_kps[:, 1], actual_kps[:, 0], s=0.1)
            # plt.scatter(data1_coords[:, 1], data1_coords[:, 0], s=0.1)
            # plt.scatter(expected_kps[:, 1], expected_kps[:, 0], s=0.1)
            # plt.xlim(0, 1024)
            # plt.ylim(0, 1024)
            # plt.show()
            
            data_out = MotionData(expected_kps, actual_kps, expected_features, actual_features)
            
            self._server.transmit(data_out)

            self._count += 1

            if self._count % 1000 == 0:
                print("Generated " + f"{self._count/self._output_size:.0%}" + " of synthetic motion data")