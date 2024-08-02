import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt

from astronet_utils import ProjectionUtils
from feature_matcher.backends.criteria import LessThan
from feature_matcher.backends.metrics import L2
from benchmark.statistics import Statistics

class VisualOdometry:
    def __init__(self, matcher):
        self._matcher = matcher
        self._keypoints_metric = L2
        self._keypoints_criterion = LessThan(1.0)

    def minimal_set(self, length, count=8):
        set_size = count
        
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
            
            #F, mask = cv2.findFundamentalMat(coords_prev, coords_next, method=cv2.FM_RANSAC, ransacReprojThreshold=0.5)
            E, mask = cv2.findEssentialMat(coords_prev, coords_next, K, method=cv2.RANSAC, threshold=1)
            #E = K.T @ F @ K

            _, R, t, _ = cv2.recoverPose(E[:3, :3], coords_prev, coords_next, K, mask)

            return R, t
        else:
            return None
        
    def recover_poses(self, idx, sampled_R, sampled_t, prev, next, E, K):
        if idx*3 < E.shape[0]:
            ess = E[idx*3:idx*3+3, :3]
            #ess[0, 2] = 0
            _, R, t, _ = cv2.recoverPose(ess, prev, next, K)
            sampled_R.append(R)
            sampled_t.append(t)
    
    def get_sampled_poses(self, data, num_samples):
        prev_points_25D = torch.hstack((data.prev_points.kps, data.prev_points.depths[:, None]))
        world_points_3D = ProjectionUtils.camera2object(data.prev_points.proj, prev_points_25D)
        reproj_points_25D = ProjectionUtils.object2camera(data.next_points.proj, world_points_3D)
        reproj_points_2D = reproj_points_25D[:, 0:2]

        true_dists = self._keypoints_metric.dist(reproj_points_2D, data.next_points.kps)
        true_matches = self._keypoints_criterion.apply(true_dists)
        
        pred_dists, pred_matches = self._matcher.match(data)

        indices = torch.nonzero(pred_matches)
                        
        coords_prev = data.prev_points.kps[indices[:, 0]]
        coords_next = data.next_points.kps[indices[:, 1]]
        
        #print(f"{coords_prev}: {coords_prev - coords_next}")

        # plt.figure()
        # hist = (coords_prev - coords_next).to(dtype=torch.float).norm(p=1, dim=1).cpu()
        # plt.hist(hist[None, :], bins=50)
        # plt.show()
    
        coords_prev_cpu = coords_prev.cpu().numpy()
        coords_next_cpu = coords_next.cpu().numpy()
        
        # plt.figure()
        # plt.scatter(all_coords_prev[:, 0], all_coords_prev[:, 1])
        # plt.scatter(all_coords_next[:, 0], all_coords_next[:, 1])
        # plt.show()
        
        if len(coords_prev) < 8 or len(coords_next) < 8:
            return None

        K = data.prev_points.proj.intrinsics.K.cpu().numpy()
        #K = np.array(((1024, 0, 512), (0, 1024, 512), (0, 0, 1)))
        
        mask = self.minimal_set(len(coords_prev))

        sampled_R = []
        sampled_t = []

        for i in range(num_samples):
            np.random.shuffle(mask)
                        
            F, _ = cv2.findFundamentalMat(coords_prev_cpu[mask], coords_next_cpu[mask], method=cv2.FM_8POINT)
            #E, _ = cv2.findEssentialMat(coords_prev_cpu[mask], coords_next_cpu[mask])

            # F[1, 0] = 0
            # F[1, 1] = 0
            # F[1, 2] = 0
                        
            if F is not None:
                E = K.T @ F @ K
                
                #E[1, 0] = 0
                #E[1, 1] = 0
                #E[0, 2] = 0
                #E[1, 2] = 0
                #E[2, 2] = 0
            
                self.recover_poses(0, sampled_R, sampled_t, coords_prev_cpu[mask], coords_next_cpu[mask], E, K)
                #self.recover_poses(1, sampled_R, sampled_t, coords_prev_cpu[mask], coords_next_cpu[mask], E, K)
                #self.recover_poses(2, sampled_R, sampled_t, coords_prev_cpu[mask], coords_next_cpu[mask], E, K)
                #self.recover_poses(3, sampled_R, sampled_t, coords_prev_cpu[mask], coords_next_cpu[mask], E, K)

        return np.array(sampled_R), np.array(sampled_t)