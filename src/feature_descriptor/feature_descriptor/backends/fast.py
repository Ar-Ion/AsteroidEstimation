import cv2
import torch

from . import Backend

class FASTBackend(Backend):

    def __init__(self, client, server, size, backend_params):
        super().__init__(client, server, size, backend_params)
        self._extractor = cv2.FastFeatureDetector_create()

    def detect_features(self, image):
        (kp, dess) = self._extractor.detectAndCompute(image, None)
        return (torch.tensor(list(map(lambda x: (x.pt[1], x.pt[0]), kp)), dtype=torch.int), torch.from_numpy(dess))