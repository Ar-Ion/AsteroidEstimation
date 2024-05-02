from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import MinkowskiEngine as ME
import wandb

from benchmark.statistics import *
from astronet_msgs import BatchedMotionData
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
        self._epsilon = torch.tensor(2).sqrt()
        lr = 0.0001
        momentum = 0.9
        weight_decay=0.0
        
        # Normalization (comes from verify_dataset.py)
        #mean = 1.471
        #std = 0.0452
        #transform = lambda x: (x - mean)/std
        
        # Initialize datasets for training and validate
        self._train_frontend = train_frontend
        self._validate_frontend = validate_frontend

        train_dataset = AsteroidMotionDataset(train_frontend, train_size)
        validate_dataset = AsteroidMotionDataset(validate_frontend, validate_size)
        
        self._train_dataloader = AsteroidMotionDataset.DataLoader(train_dataset, self._batch_size)
        self._validate_dataloader = AsteroidMotionDataset.DataLoader(validate_dataset, self._batch_size)

        # CUDA configuration
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        
        print("Using compute module " + str(self._device))
        
        # Model instantiation
        model_wrapped = SuperPoint()
        
        #if torch.cuda.device_count() > 1:
        #    self._model = torch.nn.parallel.DistributedDataPaactual_featuresrallel(model_wrapped.to(self._device))
        #    print("Using multiple GPUs")
        #else:
        self._model = model_wrapped.to(self._device)
        
        # Loss function metrics
        self._coords_matcher = Matcher(L2, LowerThan(self._epsilon))
        self._features_matcher = Matcher(Cosine, MaxRatio(1.0))
        
        # Optimizer instantiation
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        
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
        
        #self._model.load_state_dict(torch.load(self._pth_path))
                
    # Main loop. Called by the ROS node. Supposedly trains the neural network.
    def loop(self):
        
        first_sample = self._validate_frontend.get_sync(0)
        
        best_f1 = 0
        
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
                loss.backward()
                self._optimizer.step()
            

            # Validation
            val_losses = []
            val_stats = []

            self._model.eval()
            
            # Validate model
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
            batch = BatchedMotionData.from_list([first_sample])
            batch_gpu = batch.to(self._device)
            batch_output = self.forward_each(batch_gpu)
            output = batch_output.retrieve(0)

            self._visualizer.plot(output.prev_kps.detach().cpu().numpy(), output.prev_features.detach().cpu().numpy())
            
            true_dists, true_matches = self._coords_matcher.match(output.proj_kps, output.next_kps)
            pred_dists, pred_matches = self._features_matcher.match(output.prev_features, output.next_features)
            
            stats = Statistics(true_dists, true_matches, pred_dists, pred_matches)

            # Saving current weights to provided file
            if avg_val_f1 > best_f1:
                torch.save(self._model.state_dict(), self._pth_path)
                print("This is the best model so far")
                best_f1 = avg_val_f1
                
            
        wandb.finish()

    ## Forward pass methods
    
    # Forwards a batch to the model
    def forward(self, batch): 
        batch_gpu = batch.to(self._device)
        batch_output = self.forward_each(batch_gpu)
        batch_loss = 0
        batch_stats = []

        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            for idx in range(self._batch_size):
                output = batch_output.retrieve(idx)

                true_dists, true_matches = self._coords_matcher.match(output.proj_kps, output.next_kps)
                pred_dists, pred_matches = self._features_matcher.match(output.prev_features, output.next_features)
                
                batch_loss += self.loss(true_dists, true_matches, pred_dists, pred_matches)
                batch_stats.append(Statistics(true_dists.detach(), true_matches.detach(), pred_dists.detach(), pred_matches.detach()))                    
                
        loss = batch_loss / self._batch_size

        return (loss, batch_stats)

    # Forwards the batch by decomposing the motion into the two sets of features and coordinates
    def forward_each(self, batch):
        prev_features_out = self.forward_sparse(batch.prev_kps, batch.prev_features)
        next_features_out = self.forward_sparse(batch.next_kps, batch.next_features)
        return BatchedMotionData(batch.prev_kps, batch.proj_kps, batch.next_kps, prev_features_out, next_features_out)
    
    # Converts the coordinates and features to a MinkowskiEngine-compliant format and forwards them to the NN
    def forward_sparse(self, input_coords, input_features):
        input = ME.SparseTensor(input_features, input_coords)
        output = self._model.forward(input)
        return output.features
    
    ## Loss function
    
    # Loss function used to train the model
    def loss(self, true_dists, true_matches, pred_dists, pred_matches):      
        
        # Add dustbin to allow unmatched feature
        
        pred_dists = torch.nn.functional.normalize(pred_dists, dim=0)
        
        weight = 1 # This parameter has little effect
        margin = 0.3
        
        # Ground truth
        matches = true_matches
        diffs = 1-true_matches

        # Hinge loss
        fn_loss = matches * torch.maximum(torch.tensor(0), 1.0 - pred_dists)
        fp_loss = diffs * torch.maximum(torch.tensor(0), pred_dists - margin)
        
        # Average
        fn_loss_avg = fn_loss.sum()
        fp_loss_avg = fp_loss.sum()

        #print(fn_loss_avg)
        #print(fp_loss_avg)
        
        # Combine
        hinge_loss = weight*fn_loss_avg + fp_loss_avg
        
        return hinge_loss.sum()# + ld*norm_soft_loss.mean()#torch.nn.functional.binary_cross_entropy_with_logits(pred_dists, true_matches)

        # normed_match_matrix = match_matrix**2

        # # torch.log(torch.nn.functional.normalize(1 + distance_matrix, dim=0)) +
        # expected_likelihood = torch.log_softmax(-distance_matrix**2/(2*sigma*sigma), dim=0)
        # predicted_likelihood = torch.log(normed_match_matrix / torch.sum(normed_match_matrix, dim=0))

        # kl_div = torch.sum(torch.exp(expected_likelihood) * (expected_likelihood - predicted_likelihood), dim=0)
        
        # return kl_div.mean()