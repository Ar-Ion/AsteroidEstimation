import torch
import numpy as np
from abc import ABC, abstractmethod
from astronet_msgs import FeatureData

class Backend(ABC):

    def __init__(self, client, server, size, backend_params):
        self._client = client
        self._server = server
        self._size = size
        self._count = 0

    def loop(self):
        while self._count < self._size:
            data = self._client.receive(blocking=True)

            camera_data = data.robot_data.cam_data[0]
            
            (coords, features) = self.detect_features(camera_data.image)
            
            coords_cpu = coords.cpu()
            features_cpu = features.cpu()
            
            depth_np = camera_data.depth[coords_cpu[:, 0], coords_cpu[:, 1]]
            
            if type(depth_np) == np.ndarray:
                depth_value = torch.from_numpy(depth_np)
            else:
                depth_value = torch.tensor(depth_np, device="cpu")
            
            depth_value = depth_value.reshape((-1, 1))

            cam_coords = torch.hstack([coords_cpu.to(dtype=torch.float), depth_value])
            
            features_data = FeatureData.RobotData.SparseCameraData(camera_data.pose, camera_data.k, cam_coords, features_cpu)
            robot_data = FeatureData.RobotData([features_data])
            env_data = FeatureData.EnvironmentData(data.env_data.pose)
            data_out = FeatureData(env_data, robot_data)

            self._server.transmit(data_out)

            self._count += 1
            
            if self._count % 100 == 0:
                print("Generated " + f"{self._count/self._size:.0%}" + " of synthetic feature data")

        print("All synthetic data has been generated")

    @abstractmethod
    def detect_features(self, image):
        pass

