import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import Backend

class KAZEBackend(Backend):

    def __init__(self):
        self._extractor = cv2.KAZE_create()

    def extract_features(self, image):
        return self._extractor.detectAndCompute(image, None)

    def get_match_norm(self):
        return cv2.NORM_L2