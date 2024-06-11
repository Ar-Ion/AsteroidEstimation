import torch

from benchmark.statistics import Statistics
from astronet_utils import ProjectionUtils, PointsUtils

from .metrics import L2
from .criteria import LessThan

class Backend:
    def __init__(self, frontend, size, matcher):
        self._frontend = frontend
        self._size = size
        
        self._keypoints_metric = L2
        self._keypoints_criterion = LessThan(2)
        self._matcher = matcher

        print("Loading GPU...")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("GPU loaded. Using compute device: " + str(self._device))

    def loop(self):
        count = 0

        stats = []
        feature_count = 0

        while count < self._size:
            data = self._frontend.receive(blocking=True)
            
            prev_points_2D = PointsUtils.to(data.prev_points, device=self._device)
            next_points_2D = PointsUtils.to(data.next_points, device=self._device)
            prev_points_25D = torch.hstack((prev_points_2D.kps, prev_points_2D.depths[:, None]))
            world_points_3D = ProjectionUtils.camera2object(prev_points_2D.proj, prev_points_25D)
            reproj_points_25D = ProjectionUtils.object2camera(next_points_2D.proj, world_points_3D)
            reproj_points_2D = reproj_points_25D[:, 0:2]

            true_dists = self._keypoints_metric.dist(reproj_points_2D, next_points_2D.kps)
            true_matches = self._keypoints_criterion.apply(true_dists)
            
            pred_dists, pred_matches = self._matcher.match(data)
                        
            stats.append(Statistics(true_dists, true_matches, pred_matches))            
            feature_count += 0.5*(data.prev_features.size(0) + data.next_features.size(0))
            
            count += 1

            if count % 100 == 0:
                print("Computed statistics for " + f"{count/self._size:.0%}" + " of the described data")

        avg_accuracy = torch.tensor(list(map(lambda x: x.accuracy(), stats))).mean()
        avg_precision = torch.tensor(list(map(lambda x: x.precision(), stats))).mean()
        avg_recall = torch.tensor(list(map(lambda x: x.recall(), stats))).mean()
        avg_f1 = torch.tensor(list(map(lambda x: x.f1(), stats))).mean()
        avg_feature_count = feature_count / self._size
        avg_true_count = torch.tensor(list(map(lambda x: x.true_count(), stats))).mean()
        avg_positive_count = torch.tensor(list(map(lambda x: x.positive_count(), stats))).mean()
        avg_pixel_error = torch.tensor(list(map(lambda x: x.pixel_error(), stats))).mean()

        print(f"Average accuracy: {avg_accuracy}")
        print(f"Average precision: {avg_precision}")
        print(f"Average recall: {avg_recall}")
        print(f"Average F1 score: {avg_f1}")
        print(f"Average feature count: {avg_feature_count}")
        print(f"Average true count: {avg_true_count}")
        print(f"Average positive count: {avg_positive_count}")
        print(f"Average pixel location error: {avg_pixel_error}")