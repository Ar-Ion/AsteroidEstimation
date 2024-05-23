import numpy as np
import quaternion
import torch

from astronet_frontends import DriveClientFrontend
from astronet_utils import ExtrinsicsUtils, IntrinsicsUtils, ProjectionUtils
from astronet_msgs import MotionData, ProjectionData

class MotionGenerator:
    def __init__(self, client, server, input_size, output_size):
        self._client = client
        self._server = server
        self._input_size = input_size
        self._output_size = output_size

        # CUDA configuration
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("Using compute module " + str(self._device))

    def get_projection(self, data):
        env_pose = data.env_data.pose
        cam_pose = data.robot_data.cam_data[0].pose
        
        env_quat = np.quaternion(*env_pose.rot)
        cam_quat = np.quaternion(*cam_pose.rot)

        rot = np.quaternion(np.sqrt(2)/2, -np.sqrt(2)/2, 0, 0) # I found this matrix empirically... Ideally, use cam_quat instead

        env2cam_trans = quaternion.rotate_vectors(cam_quat, env_pose.trans - cam_pose.trans)
        env2cam_quat = rot * env_quat

        intrinsics = IntrinsicsUtils.from_K(np.array(data.robot_data.cam_data[0].k))
        extrinsics = ExtrinsicsUtils.from_SE3_7D(env2cam_trans, env2cam_quat)

        return ProjectionData(intrinsics, extrinsics)

    def loop(self):
        
        count = 0
        
        while count < self._output_size:
            if count != 0 and count*2 % self._input_size == 0:
                # We have iterated through the whole input dataset. The data must be reshuffled.
                self._client.send_event(DriveClientFrontend.Events.RESET)
        
            data1 = self._client.receive(blocking=True)
            data2 = self._client.receive(blocking=True)
            
            proj1 = ProjectionUtils.to(self.get_projection(data1), device=self._device)
            proj2 = ProjectionUtils.to(self.get_projection(data2), device=self._device)
            
            distance1 = torch.Tensor(data1.robot_data.cam_data[0].pose.trans - data1.env_data.pose.trans).to(self._device)
            distance2 = torch.Tensor(data2.robot_data.cam_data[0].pose.trans - data2.env_data.pose.trans).to(self._device)
                        
            data1_coords = torch.Tensor(data1.robot_data.cam_data[0].coords).to(self._device)
            data1_features = torch.Tensor(data1.robot_data.cam_data[0].features).to(self._device)
            data2_coords = torch.Tensor(data2.robot_data.cam_data[0].coords).to(self._device)
            data2_features = torch.Tensor(data2.robot_data.cam_data[0].features).to(self._device)
            
            env1_coords = ProjectionUtils.camera2object(proj1, data1_coords)
            reproj1_coords = ProjectionUtils.object2camera(proj2, env1_coords)

            # Remove all features that cannot be visble on the next image
            non_nan_filter = ~torch.any(reproj1_coords.isnan(), dim=1)
            visible_filter = reproj1_coords[:, 2] < torch.norm(distance2)
            combined_filter = non_nan_filter & visible_filter

            prev_kps = data1_coords[combined_filter, 0:2].to(dtype=torch.int).cpu()
            next_kps = data2_coords[:, 0:2].to(dtype=torch.int).cpu()
            prev_features = data1_features[combined_filter].cpu()
            next_features = data2_features.cpu()
            prev_depths = data1_coords[combined_filter, 2].cpu()
            next_depths = data2_coords[:, 2].cpu()
            prev_proj = ProjectionUtils.to(proj1, device="cpu")
            next_proj = ProjectionUtils.to(proj2, device="cpu")
                        
            # plt.figure()
            # plt.scatter(prev_kps[:, 1], prev_kps[:, 0], s=0.1)
            # plt.scatter(proj_kps[:, 1], proj_kps[:, 0], s=0.1)
            # plt.scatter(next_kps[:, 1], next_kps[:, 0], s=0.1)
            # plt.xlim(0, 1024)
            # plt.ylim(0, 1024)
            # plt.show()
            
            prev_points = MotionData.PointsData(
                prev_kps,
                prev_depths,
                prev_features,
                prev_proj
            )
            
            next_points = MotionData.PointsData(
                next_kps,
                next_depths,
                next_features,
                next_proj
            )
            
            data_out = MotionData(prev_points, next_points)
            
            self._server.transmit(data_out)

            count += 1

            if count % 100 == 0:
                print("Generated " + f"{count/self._output_size:.0%}" + " of synthetic motion data")