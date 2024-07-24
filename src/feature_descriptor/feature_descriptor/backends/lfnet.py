import cv2
import torch
from matplotlib import pyplot as plt
from slam.pyslam.feature_lfnet import LfNetFeature2D

from . import Backend

class LFNetBackend(Backend):

    def __init__(self, client, server, size, backend_params):
        super().__init__(client, server, size, backend_params)
        self._extractor = LfNetFeature2D()

    def detect_features(self, image):
        (kp, dess) = self._extractor.detectAndCompute(image, None)
        return (torch.tensor(list(map(lambda x: (x.pt[1], x.pt[0]), kp)), dtype=torch.int), torch.from_numpy(dess))
