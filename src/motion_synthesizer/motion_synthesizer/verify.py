import rclpy
import torch
from matplotlib import pyplot as plt

from astronet_frontends import AsyncFrontend, factory

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("motion_synthesizer_verification_node")
    
    node.declare_parameter("size", rclpy.Parameter.Type.INTEGER)
    node.declare_parameter("mode", rclpy.Parameter.Type.STRING)

    node.declare_parameters("", [
        ("input.type", rclpy.Parameter.Type.STRING),
        ("input.path", rclpy.Parameter.Type.STRING)
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

            distance_matrix = torch.cdist(data.proj_kps.float(), data.next_kps.float())
            
            matches = torch.count_nonzero(distance_matrix < torch.tensor(2).sqrt())
            sizes = torch.tensor((distance_matrix.shape[0], distance_matrix.shape[1]))
            ratio = matches/sizes.min()
            
            if ratio < 0.3:
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
            
            if len(data.prev_features) > 1:                
                features_means.append(data.prev_features.mean())
                features_stds.append(data.prev_features.std())
                
            if len(data.next_features) > 1:
                features_means.append(data.next_features.mean())
                features_stds.append(data.next_features.std())
                
            count += 1

            if count % 100 == 0:
                print("Verified " + f"{count/size:.0%}" + " of synthetic motion data")
                                                
        print(f"Mean: {torch.tensor(features_means).mean()}")
        print(f"Std: {torch.tensor(features_stds).mean()}")
        
        plt.figure()
        plt.hist(torch.hstack(dist_stats), bins="auto")
        plt.show()

        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        frontend.stop() 
        
if __name__ == '__main__':
    main()