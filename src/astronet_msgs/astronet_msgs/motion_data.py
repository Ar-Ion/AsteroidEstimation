import torch
import MinkowskiEngine as ME

class MotionData:
    def __init__(self, prev_kps, proj_kps, next_kps, prev_features, next_features):
        self.prev_kps = prev_kps
        self.proj_kps = proj_kps
        self.next_kps = next_kps
        self.prev_features = prev_features
        self.next_features = next_features

class BatchedMotionData:
    def __init__(self, prev_kps, proj_kps, next_kps, prev_features, next_features):
        self.prev_kps = prev_kps
        self.proj_kps = proj_kps
        self.next_kps = next_kps
        self.prev_features = prev_features
        self.next_features = next_features
        
    def from_list(motion_data_list):
        collated_prev_kps = [x.prev_kps for x in motion_data_list]
        collated_proj_kps = [x.proj_kps for x in motion_data_list]
        collated_next_kps = [x.next_kps for x in motion_data_list]
        collated_prev_features = [x.prev_features for x in motion_data_list]
        collated_next_features = [x.next_features for x in motion_data_list]

        # Add batch dimension
        batched_prev_kps = ME.utils.batched_coordinates(collated_prev_kps)
        batched_proj_kps = ME.utils.batched_coordinates(collated_proj_kps)
        batched_next_kps = ME.utils.batched_coordinates(collated_next_kps)

        # Concatenate all lists
        batched_prev_features = torch.concatenate(collated_prev_features, 0)
        batched_next_features = torch.concatenate(collated_next_features, 0)

        return BatchedMotionData(batched_prev_kps, batched_proj_kps, batched_next_kps, batched_prev_features, batched_next_features)

    def to(self, device=None, dtype=None):
        prev_kps = self.prev_kps.to(device=device, dtype=dtype)
        proj_kps = self.proj_kps.to(device=device, dtype=dtype)
        next_kps = self.next_kps.to(device=device, dtype=dtype)
        prev_features = self.prev_features.to(device=device, dtype=dtype)
        next_features = self.next_features.to(device=device, dtype=dtype)

        return BatchedMotionData(prev_kps, proj_kps, next_kps, prev_features, next_features)
    
    def retrieve(self, index):
        prev_filter = self.prev_kps[:, 0] == index
        next_filter = self.next_kps[:, 0] == index
        
        prev_kps = self.prev_kps[prev_filter, 1:3]
        proj_kps = self.proj_kps[prev_filter, 1:3]
        next_kps = self.next_kps[next_filter, 1:3]
        prev_features = self.prev_features[prev_filter]
        next_features = self.next_features[next_filter]

        return MotionData(prev_kps, proj_kps, next_kps, prev_features, next_features)