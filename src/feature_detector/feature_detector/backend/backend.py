from abc import ABC, abstractmethod
import numpy as np
from astronet_msgs import FeatureData

class Backend(ABC):

    def __init__(self, client, server, size):
        self._client = client
        self._server = server
        self._size = size
        self._count = 0

    def loop(self):
        while self._count < self._size:
            data = self._client.receive(blocking=True)

            camera_data = data.robot_data.cam_data[0]
            
            (coords, features) = self.detect_features(camera_data.image)
            depth_value = camera_data.depth[coords[:, 0], coords[:, 1]]
            cam_coords = np.hstack([coords, depth_value[:, None]])
            
            features_data = FeatureData.RobotData.SparseCameraData(camera_data.pose, camera_data.k, cam_coords, features)
            robot_data = FeatureData.RobotData([features_data])
            env_data = FeatureData.EnvironmentData(data.env_data.pose)
            data_out = FeatureData(env_data, robot_data)

            self._server.transmit(data_out)

            self._count += 1

            if self._count % 1000 == 0:
                print("Generated " + f"{self._count/self._size:.0%}" + " of synthetic feature data")

        print("All synthetic data has been generated")

    @abstractmethod
    def detect_features(self, image):
        pass

