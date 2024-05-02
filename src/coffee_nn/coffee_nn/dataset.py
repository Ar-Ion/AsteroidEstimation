import threading
from torch.utils.data import Dataset, DataLoader
from astronet_frontends import DriveClientFrontend
from astronet_msgs import BatchedMotionData

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
                
        
        prev_features = data.prev_features
        next_features = data.next_features
        
        if self._transform:
            prev_features = self._transform(prev_features)
            next_features = self._transform(next_features)
            
        return data
                    
    def __len__(self):
        return self._size

    def DataLoader(dataset, batch_size, drop_last=False):
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=1, 
            collate_fn=BatchedMotionData.from_list,
            drop_last=drop_last
        )