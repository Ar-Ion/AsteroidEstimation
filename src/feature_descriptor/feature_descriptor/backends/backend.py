import torch
import time
import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from astronet_msgs import FeatureData

class Backend(ABC):

    def __init__(self, client, server, size, backend_params):
        self._client = client
        self._server = server
        self._size = size
        self._count = 0

    def loop(self):
        
        plt.figure(dpi=200)
        
        time_stats = []
                
        while self._count < self._size:
            data = self._client.receive(blocking=True)

            camera_data = data.robot_data.cam_data[0]
            
            start = time.time()
            (coords, features) = self.detect_features(camera_data.image)
            time_stats.append(time.time() - start)
            
            coords_cpu = coords.cpu()
            features_cpu = features.cpu()
                        

            # plt.clf()
            # plt.imshow(camera_data.image, cmap='gray', vmin=0, vmax=255)
            # plt.scatter(coords_cpu[:, 1], coords_cpu[:, 0], s=1, color='#00ff00')
            # plt.xlim(0, 1024)
            # plt.ylim(1024, 0)
            # plt.show()
            #plt.pause(0.1)
            # plt.savefig(f"/home/arion/AsteroidRenderDetected/{self._count:04}.png")
            
            
            
            # r_occurrences = camera_data.image
            # r_range = np.max(r_occurrences) - np.min(r_occurrences)
            # plt.figure()
            # plt.title("Distribution of reds in a sample tissue of mucinous adenocarcinoma")
            # plt.xlabel("Intensity")
            # plt.ylabel("Occurrences")
            # plt.hist(r_occurrences, int(r_range))
            # plt.show()
            
            
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
        print(f"Average description time: {np.median(time_stats)}")

    @abstractmethod
    def detect_features(self, image):
        pass

