from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import MinkowskiEngine as ME
import wandb
import gc
from itertools import chain

from feature_matcher.backends import ClassicalMatcher, LightglueMatcher, SuperglueMatcher
from feature_matcher.backends.metrics import L2, Cosine
from feature_matcher.backends.criteria import GreaterThan, LessThan, MinRatio, Intersection, MaxRatio
from benchmark.statistics import Statistics
from astronet_msgs import BatchedMotionData
from feature_descriptor.backends.coffee import COFFEEDescriptor
from .dataset import AsteroidMotionDataset
from .visualizer import Visualizer
from .losses import HingeLoss, CrossEntropyLoss

class TrainDataProvider:
    def __init__(self, frontend, size):
        self.frontend = frontend
        self.size = size

class TrainPhase:    
    def __init__(self, train_dp, validate_dp, iter_ratio, batch_size):
        iter_train_size = train_dp.size // iter_ratio
        iter_validate_size = validate_dp.size // iter_ratio
        
        train_dataset = AsteroidMotionDataset(train_dp.frontend, iter_train_size)
        validate_dataset = AsteroidMotionDataset(validate_dp.frontend, iter_validate_size)
        
        self.train_dataloader = AsteroidMotionDataset.DataLoader(train_dataset, min(batch_size, iter_train_size), evaluate=True)
        self.validate_dataloader = AsteroidMotionDataset.DataLoader(validate_dataset, min(batch_size, iter_validate_size), evaluate=True)

class Trainer:
    def __init__(self, train_frontend, validate_frontend, train_size, validate_size, descriptor_params, matcher_params):
        # Hyperparameters
        warmup_batch_size = 1 # Note to self: don't try above 16
        warmup_iter_ratio = 512
        steady_batch_size = 1 # Change to 2 for 512 desc size
        steady_iter_ratio = 1024
        
        lr = 0.001
        gamma = 0.99
        momentum = 0.9
        weight_decay = 0 # 1e-5

        # Initialize datasets for training and validate
        self._train_frontend = train_frontend
        self._validate_frontend = validate_frontend

        train_dp = TrainDataProvider(train_frontend, train_size)
        validate_dp = TrainDataProvider(validate_frontend, validate_size)
        
        self._warmup_phase = TrainPhase(train_dp, validate_dp, warmup_iter_ratio, warmup_batch_size)
        self._steady_phase = TrainPhase(train_dp, validate_dp, steady_iter_ratio, steady_batch_size)

        # CUDA configuration
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        
        print("Using compute module " + str(self._device))
        
        # Training statistics
        self._iter = 0
        self._max_iters = 2000

        # Model instantiation
        self._descriptor = COFFEEDescriptor(autoload=False, **descriptor_params)
        self._matcher = LightglueMatcher(criterion=GreaterThan(0.2), autoload=False, **matcher_params)
        
        # Train phase FSM
        self._phase = self._warmup_phase
        self._phase_transition = warmup_iter_ratio // 8 # Enter steady-state when 1/4 epoch has been processed

        ## Loss function metrics 
        self._loss_match = CrossEntropyLoss()
        self._loss_det = CrossEntropyLoss()
        self._loss_desc = HingeLoss(0.5, 0.01)
        self._keypoints_metric = L2
        self._keypoints_criterion = LessThan(1.0)
        self._matcher_warmup = ClassicalMatcher(criterion=GreaterThan(0.75), metric="Cosine") # This matcher is used to help the model converge
                
        # Optimizer instantiation
        params = chain(self._descriptor.model.parameters(), self._matcher.model.parameters())
        self._optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=gamma)
        
        # Debugging tools
        self._visualizer = Visualizer(1024)
        
        # Cloud reporting
        wandb.init(
            # set the wandb project where this run will be logged
            project="coffee",

            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "weight_decay": weight_decay, 
                "architecture": self._descriptor.__class__.__name__ + "@" + self._matcher.__class__.__name__,
                "dataset": "Synthetic",
                "epochs": self._max_iters,
                "batch_size": steady_batch_size
            }
        )
                
    # Main loop. Called by the ROS node. Supposedly trains the neural network.
    def loop(self):
        
        first_sample = self._validate_frontend.get_sync(0)
        
        best_f1 = 0
        
        while self._iter < self._max_iters:
            
            # Update train phase
            self.update_fsm()
            
            # Training
            train_losses = []
            train_stats = []

            self._descriptor.model.train()
            self._matcher.model.train()

            for batch in self._phase.train_dataloader:  
                self._optimizer.zero_grad()
                (loss, stats) = self.forward(batch)
                train_losses.append(loss.item())
                train_stats.extend(stats)
                loss.backward()
                self._optimizer.step()
                
            self._scheduler.step()

            # Validation
            val_losses = []
            val_stats = []

            self._descriptor.model.eval()
            self._matcher.model.eval()
            
            # Validate model
            for batch in self._phase.validate_dataloader:
                with torch.set_grad_enabled(False):
                    (loss, stats) = self.forward(batch)
                    val_losses.append(loss.item())
                    val_stats.extend(stats)
                    
            # Compute statistics
            avg_train_loss = torch.tensor(train_losses).mean()
            avg_val_loss = torch.tensor(val_losses).mean()
            avg_train_accuracy = torch.tensor(list(map(lambda x: x.accuracy(), train_stats))).mean()
            avg_val_accuracy = torch.tensor(list(map(lambda x: x.accuracy(), val_stats))).mean()
            avg_train_precision = torch.tensor(list(map(lambda x: x.precision(), train_stats))).mean()
            avg_val_precision = torch.tensor(list(map(lambda x: x.precision(), val_stats))).mean()
            avg_train_recall = torch.tensor(list(map(lambda x: x.recall(), train_stats))).mean()
            avg_val_recall = torch.tensor(list(map(lambda x: x.recall(), val_stats))).mean()
            avg_train_f1 = torch.tensor(list(map(lambda x: x.f1(), train_stats))).mean()
            avg_val_f1 = torch.tensor(list(map(lambda x: x.f1(), val_stats))).mean()
            avg_pixel_error = torch.tensor(list(map(lambda x: x.pixel_error(), stats))).mean()

            # Log statistics
            print(f"Iteration {self._iter} finished with train loss {avg_train_loss:.4}, val loss {avg_val_loss:.4}, val f1 {avg_val_f1:.1%}")
            
            wandb.log({
                "Train loss": avg_train_loss, 
                "Validation Loss": avg_val_loss,
                "Train accuracy": avg_train_accuracy*100,
                "Validation accuracy": avg_val_accuracy*100, 
                "Train precision": avg_train_precision*100, 
                "Validation precision": avg_val_precision*100, 
                "Train recall": avg_train_recall*100, 
                "Validation recall": avg_val_recall*100, 
                "Train F1-Score": avg_train_f1*100, 
                "Validation F1-Score": avg_val_f1*100,
                "Pixel error": avg_pixel_error
            })

            # View feature map
            # batch = BatchedMotionData.from_list([first_sample])
            # batch_gpu = batch.to(self._device)
            # batch_output = self.forward_each(batch_gpu)
            # output = batch_output.retrieve(0)

            # self._visualizer.plot(output.prev_kps.detach().cpu().numpy(), output.prev_features.detach().cpu().numpy())
            
            # true_dists, true_matches = self._coords_matcher.match(output.proj_kps, output.next_kps)
            # pred_dists, pred_matches = self._features_matcher.match(output.prev_features, output.next_features)
            
            # stats = Statistics(true_dists, true_matches, pred_dists, pred_matches)

            # Cache is useless with sparse convolutions
            #torch.cuda.empty_cache()

            self._iter += 1
            
            # Save the best model
            if avg_val_f1 > best_f1:
                self._descriptor.save()
                self._matcher.save()
                print("This is the best model so far")
                best_f1 = avg_val_f1
                
        gc.collect()
        wandb.finish()
        
    # Updates the state machine that handles the training phase
    def update_fsm(self):
        if self._iter == self._phase_transition:
            self._phase = self._steady_phase
        else:
            self._phase = self._phase
        

    ## Forward pass methods
    
    # Forwards a batch to the model
    def forward(self, batch): 
        batch_gpu = batch.to(device=self._device, dtype=torch.float)
                    
        batch_output = self.forward_each(batch_gpu)
        
        batch_loss = 0
        batch_normalization = 0

        batch_stats = []
                    
        for idx in range(batch_output.num_batches):
            output = batch_output.retrieve(idx)
          
            true_dists = self._keypoints_metric.dist(output.proj_kps, output.next_kps)
            true_matches = self._keypoints_criterion.apply(true_dists)
            
            if self._phase == self._steady_phase:
                pred_dists, pred_matches = self._matcher.match(output)
                loss, normalization = self._loss_match.loss(true_dists, true_matches, pred_dists, pred_matches)
            elif self._phase == self._warmup_phase:
                pred_dists, pred_matches = self._matcher_warmup.match(output)
                loss, normalization = self._loss_desc.loss(true_dists, true_matches, pred_dists, pred_matches)
            
            batch_loss += loss
            batch_normalization += normalization
            
            if true_matches.count_nonzero() > 0:
                batch_stats.append(Statistics(true_dists.detach(), true_matches.detach(), pred_matches.detach()))                    
            
        return (batch_loss / batch_normalization, batch_stats)

    # Forwards the batch by decomposing the motion into the two sets of features and coordinates
    def forward_each(self, batch):
        prev_out = self.forward_sparse(batch.prev_kps, batch.prev_features)
        next_out = self.forward_sparse(batch.next_kps, batch.next_features)

        batch.prev_features = prev_out
        batch.next_features = next_out
       
        return batch

    # Converts the coordinates and features to a MinkowskiEngine-compliant format and forwards them to the NN
    def forward_sparse(self, input_coords, input_features):
        input = ME.SparseTensor(input_features, input_coords)
        out = self._descriptor.model.forward(input)
        return out.features