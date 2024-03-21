import numpy as np

class Intrinsics:
    def __init__(self, k):
        self.k = k
        self.k_inv = np.linalg.inv(k)

    def apply(self, point):
        # First, apply the intrisic matrix
        ndc = np.matmul(self.k, point)
        
        # Then, normalize the coordinates by the depth
        return np.array([[ndc[0]/ndc[2]], [ndc[1]/ndc[2]], [ndc[2]]])

    def revert(self, point):
        # Let's first get the UV coordinates in the image frame (without depth)
        uv = [[point[0]], [point[1]], [1]]
        
        # Now, we transform the UV coordinates in normalized device coordinates (NDC) through the intrinsic matrix K
        ndc = np.matmul(self.k_inv, uv)
        
        # Finally, we add back the depth information by normalizing the NDC coordinates
        return ndc * point[2]