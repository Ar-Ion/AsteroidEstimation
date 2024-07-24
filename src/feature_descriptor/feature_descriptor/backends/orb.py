import cv2
import torch
from matplotlib import pyplot as plt

from . import Backend

class ORBBackend(Backend):

    def __init__(self, client, server, size, backend_params):
        super().__init__(client, server, size, backend_params)
        self._extractor = cv2.cuda_ORB.create(nfeatures=1000)

    def detect_features(self, image):
        cuImage = cv2.cuda_GpuMat()
        cuImage.upload(image)
        (kp, dess) = self._extractor.detectAndComputeAsync(cuImage, None)
        kp = self._extractor.convert(kp)
        dess = dess.download()
        return (torch.tensor(list(map(lambda x: (x.pt[1], x.pt[0]), kp)), dtype=torch.int), torch.from_numpy(dess))
