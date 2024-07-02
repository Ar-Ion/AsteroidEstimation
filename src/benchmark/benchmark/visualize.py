import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

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
        
        randoms = np.random.rand(128, 3)
        color_bases = randoms / np.reshape(np.linalg.norm(randoms, axis=1), (-1, 1))
        
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
                pred_dists, pred_matches = self._matcher.match(data)
                
                indices = torch.nonzero(pred_matches).cpu().numpy()
                        
                prev_coords = data.prev_points.kps.cpu().numpy()
                next_coords = data.next_points.kps.cpu().numpy()
                prev_features = data.prev_points.features.cpu().numpy()
                next_features = data.next_points.features.cpu().numpy()
                                
                normalized_prev_features = (prev_features - np.mean(prev_features, axis=0))/np.std(prev_features, axis=0)
                normed_prev_features = normalized_prev_features / np.reshape(np.linalg.norm(normalized_prev_features, axis=1), (-1, 1))
                prev_colors = np.clip(0.5*(1 + normed_prev_features @ color_bases), 0, 1)
                
                normalized_next_features = (next_features - np.mean(next_features, axis=0))/np.std(next_features, axis=0)
                normed_next_features = normalized_next_features / np.reshape(np.linalg.norm(normalized_next_features, axis=1), (-1, 1))
                next_colors = np.clip(0.5*(1 + normed_next_features @ color_bases), 0, 1)
                
                matched_prev_coords = prev_coords[indices[:, 0]]
                matched_next_coords = next_coords[indices[:, 1]]
                matched_colors = (prev_colors[indices[:, 0]] + next_colors[indices[:, 1]])/2
                
                fig = plt.figure(figsize=(14, 6))
                
                ax1 = plt.subplot(1, 2, 1)  
                ax1.scatter(data.prev_points.kps[:, 1].cpu(), data.prev_points.kps[:, 0].cpu(), s=0.1, color=prev_colors)
                ax1.set_xlim([0, 1024])
                ax1.set_ylim([0, 1024])
                ax1.grid()

                ax2 = plt.subplot(1, 2, 2)  
                ax2.scatter(data.next_points.kps[:, 1].cpu(), data.next_points.kps[:, 0].cpu(), s=0.1, color=next_colors)
                ax2.set_xlim([0, 1024])
                ax2.set_ylim([0, 1024])
                ax2.grid()
                
                for i in range(len(indices)):
                    prev = np.flip(matched_prev_coords[i])
                    next = np.flip(matched_next_coords[i])
                    color = np.hstack((matched_colors[i], np.array((0.2))))
                    
                    match = ConnectionPatch(xyA=prev, xyB=next, coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color=color)
                    fig.add_artist(match)
                    
                plt.show()
                     
            count += 1