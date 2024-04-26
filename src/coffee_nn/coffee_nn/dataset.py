import threading
import torch
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
from astronet_frontends import DriveClientFrontend

class AsteroidMotionDataset(Dataset):
    def __init__(self, frontend, size, transform=None):
        self._frontend = frontend
        self._size = size
        self._transform = transform
        self._count = 0
        self._lock = threading.Lock()
        
    # The index is not important as the data is shuffled by the frontend directly
    def __getitem__(self, index):
                
        data = self._frontend.receive(blocking=True)
        
        with self._lock:
            self._count += 1
            
            if self._count % self._size == 0:
                # We reached the end of the dataset. Reset index and shuffle data.
                self._frontend.send_event(DriveClientFrontend.Events.RESET)
                
        
        expected_features = data.expected_features
        actual_features = data.actual_features
        
        if self._transform:
            expected_features = self._transform(expected_features)
            actual_features = self._transform(actual_features)
            
        return (data.expected_kps, data.actual_kps, expected_features, actual_features)
                    
    def __len__(self):
        return self._size
    
    def _collate(data):
        c1, c2, f1, f2 = list(zip(*data))

        # Create batched coordinates for the SparseTensor input
        bc1 = ME.utils.batched_coordinates(c1)
        bc2 = ME.utils.batched_coordinates(c2)

        # Concatenate all lists
        bf1 = torch.concatenate(f1, 0)
        bf2 = torch.concatenate(f2, 0)

        return (bc1, bc2, bf1, bf2)
    
    def DataLoader(dataset, batch_size, drop_last=False):
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=1, 
            collate_fn=AsteroidMotionDataset._collate,
            drop_last=drop_last
        )