import numpy as np
import torch
import quaternion

class Extrinsics:
    def __init__(self, disp, quat):
        rot = quaternion.as_rotation_matrix(quat)
        self._M = torch.tensor(np.vstack((np.hstack((rot, disp[:, None])), [0, 0, 0, 1]))).float()
        self._M_inv = torch.inverse(self._M)

    def apply(self, point):
        quad = torch.einsum("ij,...j->...i", self._M, torch.hstack((point, torch.ones(point.shape[0], 1))))
        return (quad[:, 0:3]/quad[:, 3, None]).squeeze() # Always normalize by the homogenesous coordinate

    def revert(self, point):
        quad = torch.einsum("ij,...j->...i", self._M_inv, torch.hstack((point, torch.ones(point.shape[0], 1))))
        return (quad[:, 0:3]/quad[:, 3, None]).squeeze() # Always normalize by the homogeneous coordinate