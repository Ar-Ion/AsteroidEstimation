import torch
from astronet_msgs import MotionData

class Randomizer:
    def __init__(self, width, height, std_scale, std_shift):
        self._width = width
        self._height = height
        self._scale = std_scale
        self._shift = std_shift
        
    def new_domain(self):
        self._scale_x = 1 + torch.randn((1)) * self._scale
        self._scale_y = 1 + torch.randn((1)) * self._scale
        self._shift_x = torch.randn((1)) * self._shift * self._width
        self._shift_y = torch.randn((1)) * self._shift * self._height
    
    def forward(self, points):
        normalizer = torch.tensor((self._width/2, self._height/2))
        
        translate = torch.tensor((self._shift_x, self._shift_y))
        scale = torch.tensor((self._scale_x, self._scale_y))
        
        new_kps = ((points.kps - normalizer)*scale + normalizer + translate).to(dtype=torch.int)
        
        return MotionData.PointsData(
            new_kps, 
            points.depths, 
            points.features, 
            points.proj,
            points.num_batches
        )        