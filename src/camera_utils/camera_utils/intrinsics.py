import numpy as np
import torch

class Intrinsics:
    def __init__(self, k):
        self._K = torch.tensor(k).float()
        self._K_inv = torch.tensor(np.linalg.inv(k)).float()

    def apply(self, point):
        # Add a dimension to point to facilitate broadcasting
        point = point[:, :, None]
        
        # First, apply the intrisic matrix
        ndc = torch.einsum("ij,...jl->...il", self._K, point)
        
        # Then, normalize the coordinates by the depth
        return torch.hstack((ndc[:, 0]/ndc[:, 2], ndc[:, 1]/ndc[:, 2], ndc[:, 2]))

    def revert(self, point):
        # Add a dimension to point to facilitate broadcasting
        point = point[:, :, None]
        
        # Let's first get the UV coordinates in the image frame (without depth)
        uv = torch.hstack((point[:, 0], point[:, 1], torch.ones(point.shape[0], 1)))

        # Now, we transform the UV coordinates in normalized device coordinates (NDC) through the intrinsic matrix K
        ndc = torch.einsum("ij,...j->...i", self._K_inv, uv)

        # Finally, we add back the depth information by normalizing the NDC coordinates
        return ndc * point[:, 2]