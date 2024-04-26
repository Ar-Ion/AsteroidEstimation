import rclpy
import numpy as np
import torch
from matplotlib import pyplot as plt

from astronet_frontends import AsyncFrontend, DriveClientFrontend

def main(args=None):
    rclpy.init(args=args)

    size = 700

    frontend_wrapped = DriveClientFrontend("/home/arion/AsteroidImageDataset/train", size)
    frontend = AsyncFrontend(frontend_wrapped, AsyncFrontend.Modes.NO_WAIT)
    frontend.start()

    count = 0

    try:
        while count < size:
            data = frontend.receive(blocking=True)

            img = data.robot_data.cam_data[0].image
            
            plt.figure()
            plt.imshow(img, cmap="gray", vmin=0, vmax=255)
            plt.show()

            count += 1
            

        dist_stats = []
        features_means = []
        features_stds = []

        for i in range(len(dataset)):
            (c1, c2, f1, f2) = dataset[i]
            distance_matrix = torch.cdist(c1.float(), c2.float())

            plt.figure()
            plt.scatter(c1[:, 1], c1[:, 0], s=0.1)
            plt.scatter(c2[:, 1], c2[:, 0], s=0.1)
            plt.xlim((0, 1024))
            plt.ylim((0, 1024))
            plt.show()
        
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        frontend.stop() 
        
if __name__ == '__main__':
    main()