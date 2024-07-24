import torch
from .dataset import AsteroidMotionDataset

class TrainDataProvider:
    def __init__(self, frontend, size):
        self.frontend = frontend
        self.size = size

class TrainPhase:    
    def __init__(self, train_dp, validate_dp, iter_ratio, batch_size, epochs_active):
        iter_train_size = train_dp.size // iter_ratio
        iter_validate_size = validate_dp.size // iter_ratio
        
        #normalization = lambda x: (x - 1.458) / 0.2087 # This one is for 1024 input dim
        normalization = lambda x: (x - 1.458) / 0.2087 + torch.normal(torch.zeros_like(x), 0.1)
      
        train_dataset = AsteroidMotionDataset(train_dp.frontend, iter_train_size, transform=normalization)
        validate_dataset = AsteroidMotionDataset(validate_dp.frontend, iter_validate_size, transform=normalization)
        
        self.train_dataloader = AsteroidMotionDataset.DataLoader(train_dataset, min(batch_size, iter_train_size), evaluate=False)
        self.validate_dataloader = AsteroidMotionDataset.DataLoader(validate_dataset, min(batch_size, iter_validate_size), evaluate=True)
        
        self._iters_active = int(epochs_active * iter_ratio)
        self._iter = 0
        
        self._next = None
        
    def tick(self):
        self._iter += 1
    
    def set_next(self, next):
        self._next = next
        
    def transition(self):
        if self._iter >= self._iters_active:
            print("Starting a new training phase")
            return self._next
        else:
            return self
