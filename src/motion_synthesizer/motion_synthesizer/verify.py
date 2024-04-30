import rclpy
import numpy as np
import torch
from matplotlib import pyplot as plt

from astronet_frontends import AsyncFrontend, factory

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("motion_synthesizer_verification_node")
    
    node.declare_parameter("size", 64)
    node.declare_parameter("mode", "train")

    node.declare_parameters("", [
        ("input.type", "DriveClientFrontend"),
        ("input.path", "/home/arion/AsteroidMotionDataset")
    ])
    
    size = node.get_parameter("size").value
    mode = node.get_parameter("mode").value
    input_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("input").items()))
    
    frontend_wrapped = factory.instance(input_params, mode, size)
    frontend = AsyncFrontend(frontend_wrapped, AsyncFrontend.Modes.NO_WAIT)
    frontend.start()

    try:
        dist_stats = []
        features_means = []
        features_stds = []

        count = 0
        while count < size:
            data = frontend.receive(blocking=True)
            (c1, c2, f1, f2) = (data.expected_kps, data.actual_kps, data.expected_features, data.actual_features)

            distance_matrix = torch.cdist(c1.float(), c2.float())
            
            matches = np.count_nonzero(distance_matrix < 1)
            sizes = torch.tensor((distance_matrix.shape[0], distance_matrix.shape[1]))
            ratio = matches/sizes.min()
            
            if ratio < 0.2:
                print(f"The data at index {count} doesn't make sense")

            #plt.figure()
            #plt.scatter(c1[:, 1], c1[:, 0], s=0.1)
            #plt.scatter(c2[:, 1], c2[:, 0], s=0.1)
            #plt.xlim((0, 1024))
            #plt.ylim((0, 1024))
            #plt.show()

            if len(distance_matrix) > 0:
                shortest_distance = distance_matrix.min(dim=0).values
                dist_stats.append(shortest_distance)
            
            if len(f1) > 1:                
                features_means.append(f1.mean())
                features_stds.append(f1.std())
                
            if len(f2) > 1:
                features_means.append(f2.mean())
                features_stds.append(f2.std())
                
            count += 1

            if count % 100 == 0:
                print("Verified " + f"{count/size:.0%}" + " of synthetic motion data")
                                                
        plt.figure()
        plt.hist(np.hstack(dist_stats), bins="auto")
        plt.show()
                
        print(f"Mean: {np.mean(features_means)}")
        print(f"Std: {np.std(features_stds)}")
        
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        frontend.stop() 
        
if __name__ == '__main__':
    main()