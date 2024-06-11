import torch
import MinkowskiEngine as ME
from coffee_nn.models.filters import SparseFilter
from coffee_nn.common import MotionProcessingModel

class COFFEEFilter(MotionProcessingModel):
    def __init__(self, model_path, autoload=True):
        super().__init__(SparseFilter(), model_path, autoload)
        
        