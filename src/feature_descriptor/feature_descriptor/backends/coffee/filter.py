import torch
import MinkowskiEngine as ME
from coffee_nn.models.filters import SparseFilter
from coffee_nn.common import MotionProcessingModel
from astronet_utils import PointsUtils

class COFFEEFilter(MotionProcessingModel):
    
    THRESHOLD = 0.8
    
    def __init__(self, model_path, autoload=True):
        super().__init__(SparseFilter(), model_path, autoload)
        
    def apply_points(self, points_data):
        with torch.set_grad_enabled(False):
            out_features = self.forward_sparse(points_data.kps, points_data.features)
            points_data.features = torch.nn.functional.sigmoid(out_features)
            out_points = PointsUtils.stash(points_data, lambda x: x < COFFEEFilter.THRESHOLD)      
        
        return out_points
    
    def apply(self, coords, features):
        with torch.set_grad_enabled(False):
            out_features = self.forward_sparse(coords, features).squeeze()
            features = torch.nn.functional.sigmoid(out_features)
            filtered_coords = coords[features >= COFFEEFilter.THRESHOLD]
            filtered_features = features[features >= COFFEEFilter.THRESHOLD, None]
        
        return (filtered_coords, filtered_features)