import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from slam.pyslam.feature_disk import DiskFeature2D

from . import Backend

class DiskBackend(Backend):

    def __init__(self, client, server, size, backend_params):
        super().__init__(client, server, size, backend_params)
        self._extractor = DiskFeature2D()

    def detect_features(self, image):
        image = image[:, :, None]
        (kp, dess) = self._extractor.detectAndCompute(np.concatenate((image, image, image), axis=2), None)
        return (torch.tensor(list(map(lambda x: (x.pt[1], x.pt[0]), kp)), dtype=torch.int), torch.from_numpy(dess))
