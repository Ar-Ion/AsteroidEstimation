import torch
import MinkowskiEngine as ME

from astronet_msgs.motion_data import MotionData

from .projection import ProjectionUtils
from .abstractions import Batchable, Torchable, Chunkable

class PointsUtils(Batchable, Torchable, Chunkable):
    # Functionalities
    def stash(points_data, predicate):
        filter = ~predicate(points_data.features).squeeze()
        
        kps = points_data.kps[filter]
        depths = points_data.depths[filter]
        features = points_data.features[filter]
        proj = points_data.proj
        
        return MotionData.PointsData(
            kps, 
            depths, 
            features, 
            proj,
            points_data.num_batches
        )
    
    # Compliance with Torchable abstraction
    def to(points_data, device=None, dtype=None):
        kps = points_data.kps.to(device=device)
        depths = points_data.depths.to(device=device)
        features = points_data.features.to(device=device, dtype=dtype)
        proj = ProjectionUtils.to(points_data.proj, device=device, dtype=dtype)

        return MotionData.PointsData(
            kps, 
            depths, 
            features, 
            proj,
            points_data.num_batches
        )
        
    def detach(points_data):
        kps = points_data.kps.detach()
        depths = points_data.depths.detach()
        features = points_data.features.detach()
        proj = ProjectionUtils.detach(points_data.proj)

        return MotionData.PointsData(
            kps, 
            depths, 
            features, 
            proj,
            points_data.num_batches
        )
        
    # Compliance with Batchable abstraction
    def batched(points_data_list):
        num_batches = len(points_data_list)
        
        collated_kps = [x.kps for x in points_data_list]
        collated_depths = [x.depths for x in points_data_list]
        collated_features = [x.features for x in points_data_list]
        collated_proj = [x.proj for x in points_data_list]

        # Add batch dimension
        batched_kps = ME.utils.batched_coordinates(collated_kps)

        # Concatenate all lists
        batched_depths = torch.concatenate(collated_depths, 0)
        batched_features = torch.concatenate(collated_features, 0)
        batched_proj = ProjectionUtils.batched(collated_proj)
        
        return MotionData.PointsData(batched_kps, batched_depths, batched_features, batched_proj, num_batches)
        
    def retrieve(points_data, index):
        filter = points_data.kps[:, 0] == index
        
        kps = points_data.kps[filter, 1:3]
        depths = points_data.depths[filter]
        features = points_data.features[filter]
        proj = ProjectionUtils.retrieve(points_data.proj, index)

        return MotionData.PointsData(kps, depths, features, proj)
        
    # Compliance with Chunkable abstraction
    def create_chunks(points_data, in_dim, out_dim):
        
        assert points_data.num_batches == -1 # Otherwise, the data is already batched
        
        chunks_per_row = -(-in_dim//out_dim) # Equivalent to ceil(in_dim/out_dim) without importing np or math
        num_chunks = chunks_per_row**2
        
        chunk_ids = (points_data.kps[:, 0] // out_dim) + (points_data.kps[:, 1] // out_dim) * chunks_per_row
        
        out_data = []
        
        for chunk_id in range(num_chunks):
            filter = (chunk_ids == chunk_id)
                        
            kps = points_data.kps[filter]
            depths = points_data.depths[filter]
            features = points_data.features[filter]
            proj = points_data.proj
            
            out_data.append(MotionData.PointsData(kps, depths, features, proj))

        return out_data
    