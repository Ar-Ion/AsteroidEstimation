import torch
import MinkowskiEngine as ME
import wandb
import gc

from feature_matcher.backends.metrics import L2
from feature_matcher.backends.criteria import LessThan
from benchmark.statistics import Statistics
from feature_descriptor.backends.coffee import COFFEEFilter
from astronet_utils import ProjectionUtils, MotionUtils

from .trainer import Trainer
from .train_utils import TrainDataProvider, TrainPhase

class FilterTrainer(Trainer):
    def __init__(self, train_frontend, validate_frontend, train_size, validate_size, filter_params):
        # Initialize datasets for training and validate
        self._train_frontend = train_frontend
        self._validate_frontend = validate_frontend

        train_dp = TrainDataProvider(train_frontend, int(train_size))
        validate_dp = TrainDataProvider(validate_frontend, int(validate_size))
        
        self._normal_phase = TrainPhase(train_dp, validate_dp, 1, 512, 100) # Active for 100 epochs
        
        # Model instantiation
        self._filter = COFFEEFilter(autoload=False, **filter_params)

        # Call parent constructor
        super().__init__([self._filter], self._normal_phase, lr=0.001)

        ## Loss function metrics 
        self._loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10))
        self._keypoints_metric = L2
        self._keypoints_criterion = LessThan(1.0)
    
    ## Forward pass methods
    # Forwards a batch to the model
    def forward(self, batch):                     
        batch_output = self.model.forward_motion(batch)
        
        batch_loss = 0
        batch_normalization = 0

        batch_stats = []
                    
        for idx in range(batch_output.num_batches):
            output = MotionUtils.retrieve(batch_output, idx)
        
            prev_points_25D = torch.hstack((output.prev_points.kps, output.prev_points.depths[:, None]))
            world_points_3D = ProjectionUtils.camera2object(output.prev_points.proj, prev_points_25D)
            reproj_points_25D = ProjectionUtils.object2camera(output.next_points.proj, world_points_3D)
            reproj_points_2D = reproj_points_25D[:, 0:2]

            true_dists = self._keypoints_metric.dist(reproj_points_2D, output.next_points.kps)
            true_matches = self._keypoints_criterion.apply(true_dists)
            
            valid_kps = torch.any(true_matches, dim=1).to(dtype=torch.float)
            model_output = output.prev_points.features.squeeze()

            model_results = (torch.nn.functional.sigmoid(model_output) > 0.9).to(dtype=torch.int)    

            loss = self._loss(model_output, valid_kps)
            
            batch_loss += loss
            batch_normalization += len(valid_kps)
            
            if true_matches.count_nonzero() > 0:
                batch_stats.append(Statistics(true_dists.min(dim=1).values, valid_kps, model_results.detach()))                    
            
        return (batch_loss / batch_normalization, batch_stats)