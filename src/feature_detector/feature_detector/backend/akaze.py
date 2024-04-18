import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import Backend

class AKAZEBackend(Backend):

    def __init__(self):
        self._extractor = cv2.AKAZE_create()

    def detect_features(self, image):
        return self._extractor.detect(image)