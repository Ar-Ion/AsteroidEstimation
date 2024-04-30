import cv2
import torch

from . import Backend

class AKAZEBackend(Backend):

    def __init__(self, client, server, size, backend_params):
        super().__init__(client, server, size, backend_params)
        self._extractor = cv2.AKAZE_create()

    def detect_features(self, image):
        kps, dess = self._extractor.detectAndCompute(image, None)
        coords = torch.tensor(map(kps, lambda x: x.pt))
        return (coords, dess)