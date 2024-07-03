import MinkowskiEngine as ME
import torch
import copy

from .hardware import GPU

class Model:
    def __init__(self, gpu, wrapped_model, model_path, autoload=True):
        self._model_path = model_path
                
        # Load GPU 
        if gpu != None:
            self._gpu = gpu
        else:
            self._gpu = GPU("cuda:0")

        # Model instantiation
        cuda_model = wrapped_model.to(self._gpu.device)

        if gpu.ddp:
            ddp_model = torch.nn.parallel.DistributedDataParallel(cuda_model, device_ids=[self._gpu.device])
            sync_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(ddp_model)
            sync_model._set_static_graph()
            self.model = sync_model
        else:
            self.model = cuda_model
        
        self.model.eval()
        
        if autoload:
            print(f"Loading model checkpoint for {self.model.__class__.__name__}...")
            self.load()
            
    def load(self):
        self.model.load_state_dict(torch.load(self._model_path))
        
    def save(self):
        torch.save(self.model.state_dict(), self._model_path)
        
    
# Just defines some helper forward functions to be used with the MinkowskiEngine
class MotionProcessingModel(Model):
    # Forwards the batch by decomposing the motion into the two sets of features and coordinates
    def forward_motion(self, motion):
        prev_out = self.forward_sparse(motion.prev_points.kps, motion.prev_points.features)
        next_out = self.forward_sparse(motion.next_points.kps, motion.next_points.features)
                
        motion.prev_points.features = prev_out       
        motion.next_points.features = next_out
        
        return motion

    # Converts the coordinates and features to a MinkowskiEngine-compliant format and forwards them to the NN
    def forward_sparse(self, input_coords, input_features):
        input = ME.SparseTensor(input_features, input_coords)
        out = self.model.forward(input)
        return out.features
        