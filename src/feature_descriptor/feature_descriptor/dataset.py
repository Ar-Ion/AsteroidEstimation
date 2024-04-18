import torch
from torch.utils.data import Dataset
import MinkowskiEngine as ME
from astronet_frontends import DriveClientFrontend

class AsteroidMotionDataset(Dataset):
    def __init__(self, frontend, size):
        self._frontend = frontend
        self._size = size
        
    # The index is not important as the data is shuffled by the frontend directly
    def __getitem__(self, index):
        
        data = self._frontend.receive(blocking=True)
        
        if index == self._size-1:
            # We reached the end of the dataset. Reset index and shuffle data.
            self._frontend.send_event(DriveClientFrontend.Events.RESET)
            
        return (data.expected_kps[:, 0:2], data.actual_kps, data.expected_features, data.actual_features)
                    
    def __len__(self):
        return self._size
    
    def collate(data):
        c1, c2, f1, f2 = list(zip(*data))

        # Create batched coordinates for the SparseTensor input
        bc1 = ME.utils.batched_coordinates(c1)
        bc2 = ME.utils.batched_coordinates(c2)

        # Concatenate all lists
        bf1 = torch.concatenate(f1, 0)
        bf2 = torch.concatenate(f2, 0)

        return (bc1, bc2, bf1, bf2)