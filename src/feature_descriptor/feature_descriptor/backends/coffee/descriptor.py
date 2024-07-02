from coffee_nn.hardware import GPU
from coffee_nn.common import MotionProcessingModel
from coffee_nn.models.descriptors import SparseSuperPoint

class COFFEEDescriptor(MotionProcessingModel):
    def __init__(self, model_path, autoload=True, gpu=None):
        super().__init__(gpu, SparseSuperPoint(), model_path, autoload)