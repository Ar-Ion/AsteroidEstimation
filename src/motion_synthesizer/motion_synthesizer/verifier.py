import torch
from matplotlib import pyplot as plt

from astronet_utils import ProjectionUtils, PointsUtils
from feature_matcher.backends.criteria import MinRatio, LessThan, Intersection

class Verifier:
    def __init__(self, frontend, size):
        self._frontend = frontend
        self._size = size
        
        # CUDA configuration
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("Using compute module " + str(self._device))
    
    def loop(self):
        dist_stats = []
        features_means = []
        features_stds = []
        bias = []

        count = 0
        
        crit = Intersection(MinRatio(1.0), LessThan(1))
        
        while count < self._size:
            data = self._frontend.receive(blocking=True)
            
            prev_points_2D = PointsUtils.to(data.prev_points, device=self._device)
            next_points_2D = PointsUtils.to(data.next_points, device=self._device)
            prev_points_25D = torch.hstack((prev_points_2D.kps, prev_points_2D.depths[:, None]))
            world_points_3D = ProjectionUtils.camera2object(prev_points_2D.proj, prev_points_25D)
            reproj_points_25D = ProjectionUtils.object2camera(next_points_2D.proj, world_points_3D)
            reproj_points_2D = reproj_points_25D[:, 0:2]

            distance_matrix = (reproj_points_2D[:, None, :] - next_points_2D.kps[None, :, :]).squeeze()
            filter = crit.apply(distance_matrix.norm(dim=2)).to(dtype=torch.bool)
                    
            matches = torch.count_nonzero(filter)
            sizes = torch.tensor((distance_matrix.shape[0], distance_matrix.shape[1]))
            ratio = matches/sizes.min()
            
            bias.append((distance_matrix[filter]).mean(dim=0))

            if ratio < 0.3:
                print(f"The data at index {count} doesn't make sense")

                # plt.figure()
                # plt.scatter(prev_points_2D.kps[:, 1].cpu(), prev_points_2D.kps[:, 0].cpu(), s=1)
                # plt.scatter(reproj_points_2D[:, 1].cpu(), reproj_points_2D[:, 0].cpu(), s=1)
                # plt.scatter(next_points_2D.kps[:, 1].cpu(), next_points_2D.kps[:, 0].cpu(), s=1)
                # plt.xlim((0, 1024))
                # plt.ylim((0, 1024))
                # plt.show()

            if len(distance_matrix) > 0:
                shortest_distance = distance_matrix.min(dim=0).values
                dist_stats.append(shortest_distance)
            
            if len(data.prev_points.features) > 1:                
                features_means.append(data.prev_points.features.mean())
                features_stds.append(data.prev_points.features.std())
                
            if len(data.next_points.features) > 1:
                features_means.append(data.next_points.features.mean())
                features_stds.append(data.next_points.features.std())
                
            count += 1

            if count % 100 == 0:
                print("Verified " + f"{count/self._size:.0%}" + " of synthetic motion data")
                                                
        print(f"Mean: {torch.tensor(features_means).mean()}")
        print(f"Std: {torch.tensor(features_stds).mean()}")
        print(f"Bias: {torch.vstack(bias).mean(dim=0)}")

        plt.figure()
        plt.hist(torch.vstack(dist_stats).cpu(), bins="auto")
        plt.show()