import numpy as np
import torch

class Intrinsics:
    def __init__(self, k):
        self.k = torch.tensor(k).float()
        self.k_inv = torch.tensor(np.linalg.inv(k)).float()

    def apply(self, point):
        # First, apply the intrisic matrix
        ndc = self.k @ point
        
        # Then, normalize the coordinates by the depth
        return torch.vstack((ndc[0]/ndc[2], ndc[1]/ndc[2], ndc[2]))

    def revert(self, point):
        # Let's first get the UV coordinates in the image frame (without depth)
        uv = torch.vstack((point[:, 0], point[:, 1], torch.ones((point.shape[0]))))

        # Now, we transform the UV coordinates in normalized device coordinates (NDC) through the intrinsic matrix K
        ndc = self.k_inv @ uv

        # Finally, we add back the depth information by normalizing the NDC coordinates
        return ndc * point[:, 2]