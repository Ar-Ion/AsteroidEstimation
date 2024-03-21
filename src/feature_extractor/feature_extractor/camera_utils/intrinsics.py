import numpy as np

class Intrinsics:
    def __init__(self, k):
        self.k = k
        self.k_inv = np.linalg.inv(k)

    def apply(self, point):
        return np.matmul(self.k, point)

    def revert(self, point):
        return np.matmul(self.k_inv, point)