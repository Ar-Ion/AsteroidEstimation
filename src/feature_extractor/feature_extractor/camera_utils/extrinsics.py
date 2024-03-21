import numpy as np
import quaternion

class Extrinsics:
    def __init__(self, disp, quat):
        self.disp = disp
        self.rot = quaternion.as_rotation_matrix(quat)

    def apply(self, point):
        return np.matmul(self.rot, point) + self.disp

    def revert(self, point):
        return np.matmul(self.rot.transpose(), point - self.disp)