import torch
import numpy as np
from matplotlib import pyplot as plt

from feature_matcher.backends.criteria import MatchCriterion
from feature_matcher.backends.metrics import MatchMetric
from feature_matcher.backends import Matcher
from astronet_utils import MotionUtils, PointsUtils, ProjectionUtils
from astronet_frontends import DriveClientFrontend

from .statistics import Statistics

class Visualize:
    def __init__(self, frontend, size, config):
        self._frontend = frontend
        self._size = size
       
        keypoints_criterion_args = config["keypoints.criterion_args"]
        features_criterion_args = config["features.criterion_args"]
        matcher_args = config["features.matcher_args"]
        
        if not keypoints_criterion_args:
            keypoints_criterion_args = []
            
        if not features_criterion_args:
            features_criterion_args = []
            
        if not matcher_args:
            matcher_args = []

        self._keypoints_metric = MatchMetric.instance(config["keypoints.metric"])
        self._keypoints_criterion = MatchCriterion.instance(config["keypoints.criterion"], *keypoints_criterion_args)
        self._matcher = Matcher.instance(config["features.matcher"], *matcher_args)
        features_criterion = MatchCriterion.instance(config["features.criterion"], *features_criterion_args)
        self._matcher.set_criterion(features_criterion)

        print("Loading GPU...")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("GPU loaded. Using compute device: " + str(self._device))

    def loop(self):
        
        count = 0

        while count < self._size:
            input = self._frontend.receive(blocking=True)
            
            chunks = MotionUtils.create_chunks(input, 1024, 1024)                        
            valid_chunks = list(filter(MotionUtils.is_valid, chunks))
            valid_batch = MotionUtils.batched(valid_chunks)
            valid_batch_gpu = MotionUtils.to(valid_batch, device=self._device)
            
            plt.clf()
            plt.xlim((0, 1024))
            plt.ylim((0, 1024))
                                
            for idx in range(valid_batch_gpu.num_batches):
                data = MotionUtils.retrieve(valid_batch_gpu, idx)
            
                prev_points_25D = torch.hstack((data.prev_points.kps, data.prev_points.depths[:, None]))
                world_points_3D = ProjectionUtils.camera2object(data.prev_points.proj, prev_points_25D)
                reproj_points_25D = ProjectionUtils.object2camera(data.next_points.proj, world_points_3D)
                reproj_points_2D = reproj_points_25D[:, 0:2]

                true_dists = self._keypoints_metric.dist(reproj_points_2D, data.next_points.kps)
                true_matches = self._keypoints_criterion.apply(true_dists)
                pred_dists, pred_matches = self._matcher.match(data)
                
                indices = torch.nonzero(pred_matches).cpu().numpy()
                        
                all_coords_prev = data.prev_points.kps.cpu().numpy()
                all_coords_next = data.next_points.kps.cpu().numpy()
                                
                coords_prev = all_coords_prev[indices[:, 0]]
                coords_next = all_coords_next[indices[:, 1]]     
                
                plt.figure()  
                                
                for i in range(len(indices)):
                    prev = coords_prev[i]
                    next = coords_next[i]                                                            
                    plt.plot([prev[1], next[1]], [prev[0], next[0]])
                    plt.scatter(prev_points_25D[:, 1].cpu(), prev_points_25D[:, 0].cpu(), s=0.1)
                    plt.scatter(reproj_points_25D[:, 1].cpu(), reproj_points_25D[:, 0].cpu(), s=0.1)

                plt.show()
                     
            count += 1