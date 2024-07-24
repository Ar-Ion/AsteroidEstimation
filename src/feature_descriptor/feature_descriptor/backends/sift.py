import cv2
import torch
import numpy as np
import cudasift

from . import Backend

class SIFTBackend(Backend):

    def __init__(self, client, server, size, backend_params):
        super().__init__(client, server, size, backend_params)
        self._data = cudasift.PySiftData(32768)

    def detect_features(self, image):
                
        cudasift.ExtractKeypoints(
            image, 
            self._data,
            numOctaves = 5, 
            initBlur = 1.6,
            thresh = 3.5,
            lowestScale = 0, 
            upScale = True
        )
        
        df, keypoints = self._data.to_data_frame()

        return (torch.tensor((df[["ypos", "xpos"]].to_numpy()), dtype=torch.int), torch.from_numpy(keypoints))