import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import Backend

class SIFTBackend(Backend):

    def __init__(self, client, server, size, backend_params):
        super().__init__(client, server, size, backend_params)
        self._extractor = cv2.SIFT_create()

    def detect_features(self, image):
        (kp, dess) = self._extractor.detectAndCompute(image, None)
        return (np.array(list(map(lambda x: x.pt, kp)), dtype=np.uint16), dess)