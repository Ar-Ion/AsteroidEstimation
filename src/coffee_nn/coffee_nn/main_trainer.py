import torch
import MinkowskiEngine as ME

from feature_matcher.backends import ClassicalMatcher, LightglueMatcher, SuperglueMatcher
from feature_matcher.backends.metrics import L2, Cosine
from feature_matcher.backends.criteria import GreaterThan, LessThan, MinRatio, Intersection, MaxRatio
from benchmark.statistics import Statistics
from feature_descriptor.backends.coffee import COFFEEDescriptor, COFFEEFilter
from astronet_utils import ProjectionUtils, MotionUtils, PointsUtils

from .trainer import Trainer
from .losses import HingeLoss, CrossEntropyLoss
from .train_utils import TrainDataProvider, TrainPhase

class MainTrainer(Trainer):
    def __init__(self, train_frontend, validate_frontend, train_size, validate_size, descriptor_params, matcher_params, filter_params):
        # Initialize datasets for training and validate
        self._train_frontend = train_frontend
        self._validate_frontend = validate_frontend

        train_dp = TrainDataProvider(train_frontend, int(train_size))
        validate_dp = TrainDataProvider(validate_frontend, int(validate_size))
        
        self._train_coarse_descriptor_phase = TrainPhase(train_dp, validate_dp, 8, 256, 20) # Active for 1/8 epoch
        self._train_coarse_matcher_phase = TrainPhase(train_dp, validate_dp, 32, 64, 2) # Active for 1/8 epoch
        self._train_fine_phase = TrainPhase(train_dp, validate_dp, 32, 64, 100) # Active for 10 epochs

        self._train_coarse_descriptor_phase.set_next(self._train_coarse_matcher_phase)
        self._train_coarse_matcher_phase.set_next(self._train_fine_phase)
        self._train_fine_phase.set_next(None)
        
        # Model instantiation
        self._descriptor = COFFEEDescriptor(autoload=False, **descriptor_params)
        self._matcher = LightglueMatcher(criterion=GreaterThan(0.2), autoload=False, **matcher_params)
        self._filter = COFFEEFilter(autoload=True, **filter_params)

        # Call parent constructor
        super().__init__([self._descriptor, self._matcher], self._train_coarse_descriptor_phase)

        ## Loss function metrics 
        self._loss_match = CrossEntropyLoss()
        self._loss_desc = HingeLoss(0.5, 0.01)
        self._keypoints_metric = L2
        self._keypoints_criterion = LessThan(1.0)
        self._matcher_warmup = ClassicalMatcher(criterion=GreaterThan(0.5), metric="Cosine") # This matcher is used to help the model converge
        
    ## Forward pass methods
    # Forwards a batch to the model
    def forward(self, batch):         
        batch.prev_points = self._filter.apply(batch.prev_points)
        batch.next_points = self._filter.apply(batch.next_points)
        batch_output = self._descriptor.forward_motion(batch)
        
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
            
            if self._phase == self._train_coarse_descriptor_phase:
                pred_dists, pred_matches = self._matcher_warmup.match(output)
                loss, normalization = self._loss_desc.loss(true_dists, true_matches, pred_dists, pred_matches)
            elif self._phase == self._train_coarse_matcher_phase:
                detached_output = MotionUtils.detach(output)
                pred_dists, pred_matches = self._matcher.match(detached_output)
                loss, normalization = self._loss_match.loss(true_dists, true_matches, pred_dists, pred_matches)
            elif self._phase == self._train_fine_phase:
                pred_dists, pred_matches = self._matcher.match(output)
                loss, normalization = self._loss_match.loss(true_dists, true_matches, pred_dists, pred_matches)

            batch_loss += loss
            batch_normalization += normalization
            
            if true_matches.count_nonzero() > 0:
                batch_stats.append(Statistics(true_dists.detach(), true_matches.detach(), pred_matches.detach()))                    
            
        return (batch_loss / batch_normalization, batch_stats)