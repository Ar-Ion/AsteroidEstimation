import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import Backend

class SURFBackend(Backend):

    def __init__(self):
        self._extractor = cv2.xfeatures2d.SURF_create(400)

    def detect_features(self, image):
        return self._extractor.detect(image)