import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import Backend

class ORBBackend(Backend):

    def __init__(self, client, server, size, backend_params):
        super().__init__(client, server, size, backend_params)
        self._extractor = cv2.ORB_create()

    def detect_features(self, image):
        return self._extractor.detect(image)