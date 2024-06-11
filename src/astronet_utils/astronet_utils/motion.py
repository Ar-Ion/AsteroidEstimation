import torch
import MinkowskiEngine as ME

from astronet_msgs import MotionData
from .points import PointsUtils
from .abstractions import Batchable, Torchable, Chunkable

class MotionUtils(Batchable, Torchable, Chunkable):
    # Filtering functionalities
    def stash(motion_data, predicate):
        prev_points = PointsUtils.stash(motion_data.prev_points, predicate)
        next_points = PointsUtils.stash(motion_data.next_points, predicate)
        return MotionData(prev_points, next_points, num_batches=motion_data.num_batches)
    
    # Two features are matchables iif their distance is less than one and they are mutual nearest neighbours
    def stash_unmatchables(motion_data):
        assert motion_data.num_batches == -1 # Unsupported if data is already batched
        
        dist_matrix = torch.cdist(motion_data.prev_points.kps.float(), motion_data.next_points.kps.float())
       
        match_rows = torch.argmin(dist_matrix, dim=0)
        match_cols = torch.argmin(dist_matrix, dim=1)

        keep_rows = match_cols[match_rows] == torch.arange(0, match_rows.shape[0])
        keep_cols = match_rows[match_cols] == torch.arange(0, match_cols.shape[0])
        
        prev_points = motion_data.prev_points[keep_rows]
        next_points = motion_data.next_points[keep_cols]
        
        return MotionData(prev_points, next_points)

    # Checks if the match matrix is at least 256x256, for better stability of the matching algorithm
    def is_valid(motion_data):
        return min(len(motion_data.prev_points.features), len(motion_data.next_points.features)) > 16

    # Compliance with Torchable abstraction
    def to(motion_data, device=None, dtype=None):
        prev_points = PointsUtils.to(motion_data.prev_points, device=device, dtype=dtype)
        next_points = PointsUtils.to(motion_data.next_points, device=device, dtype=dtype)
        return MotionData(prev_points, next_points, num_batches=motion_data.num_batches)

    def detach(motion_data):
        prev_points = PointsUtils.detach(motion_data.prev_points)
        next_points = PointsUtils.detach(motion_data.next_points)
        return MotionData(prev_points, next_points, num_batches=motion_data.num_batches)
    
    # Compliance with Batchable abstraction
    def batched(motion_data_list):
        num_batches = len(motion_data_list)
        
        collated_prev_points = [x.prev_points for x in motion_data_list]
        collated_next_points = [x.next_points for x in motion_data_list]
        
        batched_prev_points = PointsUtils.batched(collated_prev_points)
        batched_next_points = PointsUtils.batched(collated_next_points)
        
        return MotionData(batched_prev_points, batched_next_points, num_batches)
    
    def retrieve(motion_data, index):
        prev_points = PointsUtils.retrieve(motion_data.prev_points, index)
        next_points = PointsUtils.retrieve(motion_data.next_points, index)
        return MotionData(prev_points, next_points)
    
    # Compliance with Chunkable abstraction
    # Returns a batched representation of the input motion data, where the coordinates have a batch id as first dimension
    # The first dimension is multiplied by the num_batches field, to avoid collusion when adding other batches and reducing the first dimension
    def create_chunks(motion_data, in_dim, out_dim):
        prev_out = PointsUtils.create_chunks(motion_data.prev_points, in_dim, out_dim)
        next_out = PointsUtils.create_chunks(motion_data.next_points, in_dim, out_dim)
        return list(map(lambda x: MotionData(*x), zip(prev_out, next_out)))
    