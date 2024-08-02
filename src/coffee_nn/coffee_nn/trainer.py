import torch
import wandb
import gc
from itertools import chain
from abc import ABC, abstractmethod

from astronet_utils import MotionUtils

from .logger import Logger

class Trainer(ABC):
    def __init__(self, gpu, model_containers, initial_phase, lr=0.001, gamma=1.0, weight_decay=0, load_opt_state=False):
        # Set models
        self._model_containers = model_containers

        # Training FSM
        self._phase = initial_phase
        self._iter = 0
        
        # CUDA configuration
        self._gpu = gpu

        # Logger
        if gpu.main: # Create only one logger
            names = [model_container.__class__.__name__ for model_container in model_containers]
            self._logger = Logger("@".join(names)) # Create a readable name. e.g. COFFEEDescriptor@LightglueMatcher
        else:
            self._logger = None
                
        # Optimizer instantiation
        self._params = chain(*[model_container.model.parameters() for model_container in model_containers])
        self._optimizer = torch.optim.AdamW(self._params , lr=lr)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=gamma)
        
        # Load optimizer state
        if load_opt_state:
            state = torch.load('opt_state.pth', map_location=gpu.device)
            self._optimizer.load_state_dict(state['optimizer'])
            self._scheduler.load_state_dict(state['scheduler'])

    # Main loop. Called by the ROS node. Supposedly trains the neural network.
    def loop(self):    
        
        best_f1 = 0
            
        while self._phase != None:
            # Training
            train_losses = []
            train_stats = []
            
            for model_container in self._model_containers:
                model_container.model.train()
                                        
            # Run model on training set   
            for batch in self._phase.train_dataloader: 
                
                self._optimizer.zero_grad()
                batch_gpu = MotionUtils.to(batch, device=self._gpu.device)
                
                (loss, stats) = self.forward(batch_gpu)

                train_losses.append(loss.detach().item())
                train_stats.extend(stats)
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(self._params, 10) # Life-saving black magic      
                self._optimizer.step()
                                
            self._scheduler.step()
            
            # Validation
            val_losses = []
            val_stats = []

            for model_container in self._model_containers:
                model_container.model.eval()
                
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            
            # Run model on validation set
            for batch in self._phase.validate_dataloader:
                with torch.set_grad_enabled(False):
                    batch_gpu = MotionUtils.to(batch, device=self._gpu.device)
                    (loss, stats) = self.forward(batch_gpu)
                    val_losses.append(loss.item())
                    val_stats.extend(stats)
                    
            end.record()
           
            torch.cuda.synchronize()
            
            delta = start.elapsed_time(end)

            # Cloud reporting
            if self._logger != None:
                self._logger.report(self._iter, train_losses, train_stats, val_losses, val_stats, delta)

            # Update FSM
            self._iter += 1
            self._phase.tick()
            self._phase = self._phase.transition()
            
            avg_val_f1 = torch.tensor(list(map(lambda x: x.f1(), val_stats))).nanmean()

            # Save the model periodically
            if avg_val_f1 > best_f1:
                state = {
                    'optimizer': self._optimizer.state_dict(),
                    'scheduler': self._scheduler.state_dict()
                }

                torch.save(state, 'opt_state.pth')

                for model_container in self._model_containers:
                    model_container.save()
                                        
                best_f1 = avg_val_f1
                
                print(f"Best F1 score is now {best_f1}")
                
        self.cleanup()
        
    def cleanup(self):
        gc.collect()
        wandb.finish()
        torch.cuda.empty_cache()
        
    ## Forward pass methods
    @abstractmethod
    def forward(self, batch): 
        pass