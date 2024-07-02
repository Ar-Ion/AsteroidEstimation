import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt

class VisualOdometry:
    def __init__(self, matcher):
        self._matcher = matcher

    def minimal_set(self, length, ratio=0.5):
        set_size = int(ratio*length)
        
        if length > set_size:
            return np.array([True]*set_size + [False]*(length-set_size))
        else:
            return np.array([True]*length)

        
    def compute_pose(self, data):        
        pred_dists, pred_matches = self._matcher.match(data)
        indices = torch.nonzero(pred_matches)
        
        if len(indices) > 5:
            coords_prev = data.prev_points.kps[indices[:, 0]].cpu().numpy()
            coords_next = data.next_points.kps[indices[:, 1]].cpu().numpy()

            K = data.prev_points.proj.intrinsics.K.cpu().numpy()
            
            E, mask = cv2.findEssentialMat(coords_prev, coords_next, K)
            _, R, t, _ = cv2.recoverPose(E[:3, :3], coords_prev, coords_next, K, mask)
            
            return R, t
        else:
            return None
        
    
    def get_sampled_poses(self, data, num_samples):
        pred_dists, pred_matches = self._matcher.match(data)
        indices = torch.nonzero(pred_matches).cpu().numpy()
        
        all_coords_prev = data.prev_points.kps.cpu().numpy()
        all_coords_next = data.next_points.kps.cpu().numpy()
                        
        coords_prev = all_coords_prev[indices[:, 0]]
        coords_next = all_coords_next[indices[:, 1]]
        
        # plt.figure()
        # plt.scatter(all_coords_prev[:, 0], all_coords_prev[:, 1])
        # plt.scatter(all_coords_next[:, 0], all_coords_next[:, 1])
        # plt.show()

        K = data.prev_points.proj.intrinsics.K.cpu().numpy()
        
        mask = self.minimal_set(len(coords_prev))

        sampled_R = []
        sampled_t = []

        for i in range(num_samples):
            np.random.shuffle(mask)

            E, mask2 = cv2.findEssentialMat(coords_prev[mask], coords_next[mask], K)
            
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E[:3, :3], coords_prev[mask], coords_next[mask], K, mask2)
                sampled_R.append(R)
                sampled_t.append(t)
        
        return np.array(sampled_R), np.array(sampled_t)