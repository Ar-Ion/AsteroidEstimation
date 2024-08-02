import torch

from feature_matcher.backends import LightglueMatcher, ClassicalMatcher
from feature_matcher.backends.metrics import L2
from feature_matcher.backends.criteria import GreaterThan, LessThan, MinRatio, Intersection, MaxK
from feature_matcher.backends.criteria import GreaterThan, LessThan, MinRatio, Intersection, MaxK
from benchmark.statistics import Statistics
from feature_descriptor.backends.coffee import COFFEEDescriptor
from astronet_utils import ProjectionUtils, MotionUtils

from .trainer import Trainer
from .losses import CrossEntropyLoss, KLLoss, HingeLoss
from .train_utils import TrainDataProvider, TrainPhase
from .randomizer import Randomizer

class MainTrainer(Trainer):
    def __init__(self, gpu, train_frontend, validate_frontend, train_size, validate_size, descriptor_params, matcher_params, filter_params):
        # Initialize datasets for training and validate
        self._train_frontend = train_frontend
        self._validate_frontend = validate_frontend

        train_dp = TrainDataProvider(train_frontend, int(train_size))
        validate_dp = TrainDataProvider(validate_frontend, int(validate_size))
        
        # 512 iterations per epoch, Batch size of 8, running for 100 epochs
        self._main_phase = TrainPhase(train_dp, validate_dp, 512, 4, 100) # Active for 100 epochs
        
        # Model instantiation
        self._descriptor = COFFEEDescriptor(gpu=gpu, autoload=False, **descriptor_params)
        self._matcher = LightglueMatcher(gpu=gpu, criterion=MaxK(100), autoload=False, **matcher_params) # 0.2 is arbitrary and is tuned for evaluation. No influence on training.
        # self._matcher = ClassicalMatcher(criterion=GreaterThan(0.75), metric="Cosine")

        # Domain randomization
        self._randomizer = Randomizer(1024, 1024, 0.5, 0.5)

        # Call parent constructor
        super().__init__(gpu, [self._descriptor, self._matcher], self._main_phase, lr=0.0001, gamma=0.9999)

        ## Loss function metrics 
        #self._loss_match = CrossEntropyLoss() # Cross-entropy, as specified in the LightGlue/SuperGlue paper
        self._keypoints_metric = L2 # L2 norm to match pixels in image-space
        self._loss_match = CrossEntropyLoss()

        # The MinRatio is used to force a maximum of one match per pixel, as required by the optimal transport algorithm. 
        # The LessThen force matches to be pixels less than a given distance. Technically, the number should be sqrt(2) 
        # to match all neighbouring pixels but sometimes, the surface is smooth and the shadow doesn't by exactly one pixel.
        self._keypoints_criterion = Intersection(MinRatio(1), LessThan(1))
        self._keypoints_criterion = Intersection(MinRatio(1), LessThan(1))
        
    ## Forward pass methods
    # Forwards a batch to the model
    def forward(self, batch):    
        # Descriptor forward (training)
        batch_output = self._descriptor.forward_motion(batch)


        batch_loss = 0
        batch_normalization = 0

        batch_stats = []
                            
        for idx in range(batch.num_batches):
        for idx in range(batch.num_batches):
            # Compute loss sample per sample to save memory (can be improved for sure)
            output = MotionUtils.retrieve(batch_output, idx)


            # First, concatenate the ground-truth depth to the keypoints. prev_points_25D is then the depth image, as seen from the camera.
            prev_points_25D = torch.hstack((output.prev_points.kps, output.prev_points.depths[:, None]))
            # Then, back-project from the camera space to the object space.
            world_points_3D = ProjectionUtils.camera2object(output.prev_points.proj, prev_points_25D)
            # Project from the object space with the new ground-truth rotation.
            reproj_points_25D = ProjectionUtils.object2camera(output.next_points.proj, world_points_3D)
            # Finally, remove the depth information
            reproj_points_2D = reproj_points_25D[:, 0:2]

            # Compute ground-truth distances
            true_dists = self._keypoints_metric.dist(reproj_points_2D, output.next_points.kps)
            # Compute ground-truth pixel matches
            true_matches = self._keypoints_criterion.apply(true_dists)

            self._randomizer.new_domain()
            output.prev_points = self._randomizer.forward(output.prev_points)
            output.next_points = self._randomizer.forward(output.next_points)

            # Apply LightGlue matcher...
            pred_dists, pred_matches = self._matcher.match(output)
            # ...and compute loss function
            
            loss, normalization = self._loss_match.loss(true_dists, true_matches, pred_dists, pred_matches)
                
            batch_loss += loss
            batch_normalization += normalization
            
            if true_matches.count_nonzero() > 0:
                # Avoid NaNs by ensuring there are always some valid ground-truth elements
                batch_stats.append(Statistics(true_dists.detach(), true_matches.detach(), pred_matches.detach()))                    
            
        # The normalization is only used to facilitate visualization when plotting the loss function. It doesn't affect the Adam.
        return (batch_loss / batch_normalization, batch_stats)
