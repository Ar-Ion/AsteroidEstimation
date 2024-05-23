import torch
import quaternion

from astronet_msgs import ProjectionData
from .abstractions import Batchable, Torchable

class ExtrinsicsUtils(Batchable, Torchable):
    # Constructor helpers
    def from_SE3_7D(disp, quat):
        rot = quaternion.as_rotation_matrix(quat)
        return ExtrinsicsUtils.from_SE3_12D(disp, rot)
    
    def from_SE3_12D(disp, rot):
        disp = torch.from_numpy(disp).float()
        rot = torch.from_numpy(rot).float()
        M = torch.vstack((torch.hstack((rot, disp[:, None])), torch.tensor((0, 0, 0, 1), device="cpu")))
        M_inv = torch.inverse(M)
        return ProjectionData.ExtrinsicsData(M, M_inv)
    
    # Functionalities
    def apply(extrinsics, point):
        quad = torch.einsum("ij,...j->...i", extrinsics.M, torch.hstack((point, torch.ones(point.shape[0], 1))))
        return (quad[:, 0:3]/quad[:, 3, None]).squeeze() # Always normalize by the homogenesous coordinate

    def revert(extrinsics, point):
        quad = torch.einsum("ij,...j->...i", extrinsics.M_inv, torch.hstack((point, torch.ones(point.shape[0], 1))))
        return (quad[:, 0:3]/quad[:, 3, None]).squeeze() # Always normalize by the homogeneous coordinate
    
    # Compliance with Torchable abstraction
    def to(extrinsics, device=None, dtype=None):
        M = extrinsics.M.to(device=device, dtype=dtype)
        M_inv = extrinsics.M_inv.to(device=device, dtype=dtype)
        return ProjectionData.ExtrinsicsData(M, M_inv)
    
    def detach(extrinsics):
        M = M.detach()
        M_inv = M_inv.detach()
        return ProjectionData.ExtrinsicsData(M, M_inv)
    
    # Compliance with Batchable abstraction
    def batched(extrinsics_data_list):
        num_batches = len(extrinsics_data_list)
        
        collated_M = [x.M for x in extrinsics_data_list]
        collated_M_inv = [x.M_inv for x in extrinsics_data_list]
        
        # Concatenate all lists
        batched_M = torch.stack(collated_M)
        batched_M_inv = torch.stack(collated_M_inv)
        
        return ProjectionData.ExtrinsicsData(batched_M, batched_M_inv, num_batches)
        
    def retrieve(extrinsics_data, index):        
        M = extrinsics_data.M[index, :]
        M_inv = extrinsics_data.M_inv[index, :]
        return ProjectionData.ExtrinsicsData(M, M_inv)