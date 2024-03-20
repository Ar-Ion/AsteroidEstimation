import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import Backend

class ORBBackend(Backend):

    def __init__(self):
        self._extractor = cv2.ORB_create()

    def extract_features(self, image):
        return self._extractor.detectAndCompute(image, None)

