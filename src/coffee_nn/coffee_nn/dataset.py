import threading
import random
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
            
        return data

    def __len__(self):
        return self._size
    
    def _collate(lst, evaluate):
        # chunks = []
        
        # for l in lst:
        #     chunked_data = l.to(device="cuda").create_chunks(1024, 1024)
        #     #random.shuffle(chunked_data)
        #     chunks.extend(chunked_data)
        
        # valid_chunks = list(filter(lambda x: x.is_valid(), chunks))
                        
        return BatchedMotionData.from_list(lst)
    
    def _collate_for_training(lst):
        return AsteroidMotionDataset._collate(lst, False)
    
    def _collate_for_evaluation(lst):
        return AsteroidMotionDataset._collate(lst, True)

    def DataLoader(dataset, batch_size, evaluate=True, drop_last=False):
        if evaluate:
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=8,
                collate_fn=AsteroidMotionDataset._collate_for_evaluation,
                drop_last=drop_last
            )
        else:
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=8,
                collate_fn=AsteroidMotionDataset._collate_for_training,
                drop_last=drop_last
            )