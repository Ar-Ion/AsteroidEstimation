import threading
from torch.utils.data import Dataset, DataLoader
from astronet_frontends import DriveClientFrontend
from astronet_utils import MotionUtils

class AsteroidMotionDataset(Dataset):
    def __init__(self, frontend, size, transform=None):
        self._frontend = frontend
        self._size = size
        self._transform = transform
        self._lock = threading.Lock()
        
    # The index is not important as the data is shuffled by the frontend directly
    def __getitem__(self, index):
        return self._frontend.receive(blocking=True)

    def __len__(self):
        return self._size
    
    def _collate(lst, evaluate):
        # chunks = []
        
        # for data in lst:
        #     gpu_data = MotionUtils.to(data, device="cuda")
        #     chunked_data = MotionUtils.create_chunks(gpu_data, 1024, 1024)
        #     chunks.extend(chunked_data)
        
        # valid_chunks = list(filter(MotionUtils.is_valid, chunks))
                        
        return MotionUtils.batched(lst)
    
    def _collate_for_training(lst):
        return AsteroidMotionDataset._collate(lst, False)
    
    def _collate_for_evaluation(lst):
        return AsteroidMotionDataset._collate(lst, True)

    def DataLoader(dataset, batch_size, evaluate=True, drop_last=False):
        if evaluate:
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                num_workers=8,
                shuffle=False, 
                collate_fn=AsteroidMotionDataset._collate_for_evaluation,
                drop_last=drop_last,
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                num_workers=8,
                shuffle=False, 
                collate_fn=AsteroidMotionDataset._collate_for_training,
                drop_last=drop_last,
                pin_memory=True
            )