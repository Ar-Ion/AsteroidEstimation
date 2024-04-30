from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import MinkowskiEngine as ME
import wandb

from benchmark.statistics import *
from .models.sparse import AutoEncoder, SuperPoint
from .dataset import AsteroidMotionDataset
from .visualizer import Visualizer

class Trainer:
    def __init__(self, train_frontend, validate_frontend, train_size, validate_size, output_params):
        # Setup output for model weights
        if output_params["model_type"] != "PTH":
            raise NotImplementedError("Supported outputs only include 'PTH' Pytorch models")
        
        self._pth_path = output_params["model_path"]
        
        # Hyperparameters
        self._batch_size = 16
        self._max_epochs = 500
        self._epsilon = 16/1024
        lr = 0.001
        momentum = 0.9
        weight_decay=0.0
        
        # Normalization (comes from verify_dataset.py)
        mean = 1.471
        std = 0.0452
        transform = lambda x: (x - mean)/std
        
        # Initialize datasets for training and validate
        train_dataset = AsteroidMotionDataset(train_frontend, train_size, transform=transform)
        validate_dataset = AsteroidMotionDataset(validate_frontend, validate_size, transform=transform)
        
        self._train_dataloader = AsteroidMotionDataset.DataLoader(train_dataset, self._batch_size)
        self._validate_dataloader = AsteroidMotionDataset.DataLoader(validate_dataset, self._batch_size)

        # CUDA configuration
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        
        print("Using compute module " + str(self._device))
        
        # Model instantiation
        model_wrapped = SuperPoint()
        
        #if torch.cuda.device_count() > 1:
        #    self._model = torch.nn.parallel.DistributedDataParallel(model_wrapped.to(self._device))
        #    print("Using multiple GPUs")
        #else:
        self._model = model_wrapped.to(self._device)
        
        # Loss function metrics
        self._coords_matcher = Matcher(L2, LowerThanCriterion(self._epsilon))
        self._features_matcher = Matcher(Cosine, GreaterThanCriterion(0.3))
        
        # Optimizer instantiation
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        self._scaler = GradScaler()
        
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
                "architecture": self._model.__class__.__name__,
                "dataset": "Synthetic",
                "epochs": self._max_epochs,
                "batch_size": self._batch_size,
                "epsilon": self._epsilon
            }
        )
                
    # Main loop. Called by the ROS node. Supposedly trains the neural network.
    def loop(self):
        
        best_recall = 0
        
        for epoch in range(self._max_epochs):
            # Training
            train_losses = []
            train_stats = []

            self._model.train()

            for batch in self._train_dataloader:    
                self._optimizer.zero_grad()
                (loss, stats) = self.forward(batch)
                train_losses.append(loss.item())
                train_stats.extend(stats)
                self._scaler.scale(loss).backward()
                self._scaler.step(self._optimizer)
                self._scaler.update()
            

            # Validation
            val_losses = []
            val_stats = []

            self._model.eval()

            for batch in self._validate_dataloader:
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

            # Log statistics
            print(f"(Pseudo-)epoch {epoch} finished with train loss {avg_train_loss:.4}, val loss {avg_val_loss:.4}, val f1 {avg_val_f1:.1%}")
            
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
            })
            
            # View feature map
            batch = next(iter(self._validate_dataloader))
            batch_gpu = self.to_device(batch)
            batch_output = self.forward_each(batch_gpu)
            (c1, c2, f1, f2) = self.get_sample(batch_output, 0)
            self._visualizer.plot(c2.detach().cpu().numpy(), f2.detach().cpu().numpy())
            
            # Saving current weights to provided file
            if avg_val_recall > best_recall:
                torch.save(self._model.state_dict(), self._pth_path)
                best_recall = avg_val_recall
                
            
        wandb.finish()

    ## Forward pass methods
    
    # Forwards a batch to the model
    def forward(self, batch): 
        batch_gpu = self.to_device(batch)
        batch_output = self.forward_each(batch_gpu)
        batch_loss = 0
        batch_stats = []

        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            for idx in range(self._batch_size):
                (c1, c2, f1, f2) = self.get_sample(batch_output, idx)
                                                
                true_dists, true_matches = self._coords_matcher.match(c1, c2)
                pred_dists, pred_matches = self._features_matcher.match(f1, f2)
                
                batch_loss += self.loss(true_dists, true_matches, pred_dists, pred_matches)
                batch_stats.append(Statistics(true_matches.detach(), pred_matches.detach()))                    
                
        loss = batch_loss / self._batch_size

        return (loss, batch_stats)

    # Forwards the batch by decomposing the motion into the two sets of features and coordinates
    def forward_each(self, batch):
        # "i" stands for "input", "o" stands for "output", "c" stands for "coordinate", "f" stands for "feature" 
        (ic1, ic2, if1, if2) = batch
        (oc1, of1) = self.forward_sparse(ic1, if1)
        (oc2, of2) = self.forward_sparse(ic2, if2)
        return (oc1, oc2, of1, of2)
    
    # Converts the coordinates and features to a MinkowskiEngine-compliant format and forwards them to the NN
    def forward_sparse(self, input_coords, input_features):
        input = ME.SparseTensor(input_features[:, None], input_coords)
        output = self._model.forward(input)
        output_coords = output.coordinates
        output_features = output.features
        return (output_coords, output_features)
    
    ## Loss function
    
    # Loss function used to train the model
    def loss(self, true_dists, true_matches, pred_dists, pred_matches):      
        
        # Add dustbin to allow unmatched feature
        
        pred_dists = torch.nn.functional.normalize(pred_dists, dim=0)
        
        # In a two-ended segment model, there is a 2eps probability to match a neighbouring pixel
        weight = 16 # Average number of features
        margin = 0.3
        
        # Ground truth
        matches = true_matches
        diffs = 1-true_matches

        # Hinge loss
        fn_loss = matches * torch.maximum(torch.tensor(0), 1 - pred_dists)
        fp_loss = diffs * torch.maximum(torch.tensor(0), pred_dists - margin)
        
        # Average
        fn_loss_avg = fn_loss.sum()/matches.sum()
        fp_loss_avg = fp_loss.sum()/diffs.sum()
        
        # Combine
        hinge_loss = weight*fn_loss_avg + fp_loss_avg
        
        return hinge_loss.sum()# + ld*norm_soft_loss.mean()#torch.nn.functional.binary_cross_entropy_with_logits(pred_dists, true_matches)

        # normed_match_matrix = match_matrix**2

        # # torch.log(torch.nn.functional.normalize(1 + distance_matrix, dim=0)) +
        # expected_likelihood = torch.log_softmax(-distance_matrix**2/(2*sigma*sigma), dim=0)
        # predicted_likelihood = torch.log(normed_match_matrix / torch.sum(normed_match_matrix, dim=0))

        # kl_div = torch.sum(torch.exp(expected_likelihood) * (expected_likelihood - predicted_likelihood), dim=0)
        
        # return kl_div.mean()
    
            
    ## Utility methods
    
    # Sends a given motion sample to the GPU
    def to_device(self, sample):
        c1, c2, f1, f2 = sample
        
        c1_dev = c1.to(self._device)
        c2_dev = c2.to(self._device)
        f1_dev = f1.to(self._device)
        f2_dev = f2.to(self._device)
    
        return (c1_dev, c2_dev, f1_dev, f2_dev)
    
     # Retrieves the motion sample from a given a batch index and normalizes the coordinates to the image size
    def get_sample(self, batch, idx):
        (c1, c2, f1, f2) = batch
        
        filter1 = c1[:, 0] == idx
        filter2 = c2[:, 0] == idx
        
        return (c1[filter1, 1:3], c2[filter2, 1:3], f1[filter1], f2[filter2])