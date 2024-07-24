import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from slam.pyslam.feature_contextdesc import ContextDescFeature2D

from . import Backend

class ContextDescBackend(Backend):

    def __init__(self, client, server, size, backend_params):
        super().__init__(client, server, size, backend_params)
        self._extractor = ContextDescFeature2D()

    def detect_features(self, image):
        image = image[:, :, None]
        (kp, dess) = self._extractor.detectAndCompute(np.concatenate((image, image, image), axis=2), None)
        return (torch.tensor(list(map(lambda x: (x.pt[1], x.pt[0]), kp)), dtype=torch.int), torch.from_numpy(dess))
