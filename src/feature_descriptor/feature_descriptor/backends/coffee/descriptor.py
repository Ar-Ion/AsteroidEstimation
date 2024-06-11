from coffee_nn.common import MotionProcessingModel
from coffee_nn.models.descriptors import SparseSuperPoint

class COFFEEDescriptor(MotionProcessingModel):
    def __init__(self, model_path, autoload=True):
        super().__init__(SparseSuperPoint(), model_path, autoload)