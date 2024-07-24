import cv2
import torch
import numpy as np
from akazecuda.akazecuda import AKAZE, AKAZEOptions

from . import Backend

class AKAZEBackend(Backend):

    def __init__(self, client, server, size, backend_params):
        super().__init__(client, server, size, backend_params)
        
        options = AKAZEOptions()
        options.setWidth(1024)
        options.setHeight(1024)
        
        self._extractor = AKAZE(options)

    def detect_features(self, image):
        self._extractor.Create_Nonlinear_Scale_Space(np.float32(image)/255)
        (dess, kp) = self._extractor.Compute_Descriptors()
        
        return (torch.from_numpy(kp[:, 1::-1].astype(int)), torch.from_numpy(dess))