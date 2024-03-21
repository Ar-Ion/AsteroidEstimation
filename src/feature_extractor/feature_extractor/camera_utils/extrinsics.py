import numpy as np
import quaternion

class Extrinsics:
    def __init__(self, disp, quat):
        rot = quaternion.as_rotation_matrix(quat)
        rot_inv = np.transpose(rot)
        self._M = np.vstack((np.hstack((rot, disp[:, None])), [0, 0, 0, 1]))
        self._M_inv = np.vstack((np.hstack((rot_inv, -np.matmul(rot_inv, disp[:, None]))), [0, 0, 0, 1]))

    def apply(self, point):
        quad = np.matmul(self._M, np.vstack((point, [1])))
        return quad[0:3]/quad[3] # Always normalize by the homogeneous coordinate

    def revert(self, point):
        quad = np.matmul(self._M_inv, np.vstack((point, [1])))
        return quad[0:3]/quad[3] # Always normalize by the homogeneous coordinate