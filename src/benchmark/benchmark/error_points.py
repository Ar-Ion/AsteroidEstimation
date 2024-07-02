import os
import torch
import pandas as pd
from matplotlib import pyplot as plt

from feature_matcher.backends.criteria import MatchCriterion
from feature_matcher.backends.metrics import MatchMetric
from feature_matcher.backends import Matcher
from astronet_utils import MotionUtils, PointsUtils, ProjectionUtils
from astronet_frontends import DriveClientFrontend

from .statistics import Statistics

class ErrorPoints:
    def __init__(self, frontend, size, config, output_params):
        self._frontend = frontend
        self._size = size
       
        keypoints_criterion_args = config["keypoints.criterion_args"]
        features_criterion_args = config["features.criterion_args"]
        matcher_args = config["features.matcher_args"]
        plots_dir = os.path.join(output_params["path"], "error_points")
    
        if not keypoints_criterion_args:
            keypoints_criterion_args = []
            
        if not features_criterion_args:
            features_criterion_args = []
            
        if not matcher_args:
            matcher_args = []
            
        os.makedirs(plots_dir, exist_ok=True)

        self._csv_out = os.path.join(plots_dir, "data.csv")
        self._keypoints_metric = MatchMetric.instance(config["keypoints.metric"])
        self._keypoints_criterion = MatchCriterion.instance(config["keypoints.criterion"], *keypoints_criterion_args)
        self._matcher = Matcher.instance(config["features.matcher"], *matcher_args)
        self._matcher_criterion = config["features.criterion"]
        self._matcher_criterion_args = config["features.criterion_args"]

        print("Loading GPU...")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("GPU loaded. Using compute device: " + str(self._device))

    def loop(self):                
        param_sweep = torch.linspace(0.0, 1, 50)
        error_points = []
        
        for param_idx in range(len(param_sweep)):
            
            param = param_sweep[param_idx]
            
            stats = []
            count = 0

            while count < self._size:
                input = self._frontend.receive(blocking=True)
                
                chunks = MotionUtils.create_chunks(input, 1024, 1024)                        
                valid_chunks = list(filter(MotionUtils.is_valid, chunks))
                valid_batch = MotionUtils.batched(valid_chunks)
                valid_batch_gpu = MotionUtils.to(valid_batch, device=self._device)
                                    
                for idx in range(valid_batch_gpu.num_batches):
                    data = MotionUtils.retrieve(valid_batch_gpu, idx)
                
                    prev_points_25D = torch.hstack((data.prev_points.kps, data.prev_points.depths[:, None]))
                    world_points_3D = ProjectionUtils.camera2object(data.prev_points.proj, prev_points_25D)
                    reproj_points_25D = ProjectionUtils.object2camera(data.next_points.proj, world_points_3D)
                    reproj_points_2D = reproj_points_25D[:, 0:2]

                    true_dists = self._keypoints_metric.dist(reproj_points_2D, data.next_points.kps)
                    true_matches = self._keypoints_criterion.apply(true_dists)
                    
                    features_criterion = MatchCriterion.instance(self._matcher_criterion, param)
                    self._matcher.set_criterion(features_criterion)
                    pred_dists, pred_matches = self._matcher.match(data)
                    
                    stats.append(Statistics(true_dists, true_matches, pred_matches))            
                
                count += 1

                if count % 100 == 0:
                    print("Computed statistics for " + f"{count/self._size:.0%}" + " of the described data")

            print("Error-points sweep status: " + f"{param_idx/len(param_sweep):.0%}")

            avg_error = torch.tensor(list(map(lambda x: x.pixel_error(), stats))).nanmean()
            avg_points = torch.tensor(list(map(lambda x: x.positive_count(), stats))).nanmean()
            
            error_points.append(torch.tensor((avg_points, avg_error)))
                        
        error_points_curve = torch.vstack(error_points).cpu()
        
        df = pd.DataFrame(error_points_curve)
        df.to_csv(self._csv_out)
                
        plt.figure(figsize=(8, 6))
        plt.loglog(error_points_curve[:, 0], error_points_curve[:, 1], marker='o', linestyle='-')
        plt.xlabel('Number of points')
        plt.ylabel('Error')
        plt.title('Error-Points Curve')
        plt.grid(True)
        plt.xlim([1, 1e3])
        plt.ylim([1, 1e3])
        plt.show()