import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import Backend

class FASTBackend(Backend):

    def __init__(self):
        self._extractor = cv2.FastFeatureDetector_create()

    def detect_features(self, image):
        return self._extractor.detect(image)