from astronet_msgs import ProjectionData
from .intrinsics import IntrinsicsUtils
from .extrinsics import ExtrinsicsUtils
from .abstractions import Batchable, Torchable

class ProjectionUtils(Batchable, Torchable):
    # Functionalities
    def camera2object(projection_data, point):
        return ExtrinsicsUtils.revert(projection_data.extrinsics, IntrinsicsUtils.revert(projection_data.intrinsics, point.float()))

    def object2camera(projection_data, point):
        return IntrinsicsUtils.apply(projection_data.intrinsics, ExtrinsicsUtils.apply(projection_data.extrinsics, point.float()))

    # Compliance with Torchable abstraction
    def to(projection_data, device=None, dtype=None):
        intrinsics = IntrinsicsUtils.to(projection_data.intrinsics, device=device, dtype=dtype)
        extrinsics = ExtrinsicsUtils.to(projection_data.extrinsics, device=device, dtype=dtype)
        return ProjectionData(intrinsics, extrinsics)
    
    def detach(projection_data):
        intrinsics = IntrinsicsUtils.detach(projection_data.intrinsics)
        extrinsics = ExtrinsicsUtils.detach(projection_data.extrinsics)
        return ProjectionData(intrinsics, extrinsics)
    
    # Compliance with Batchable abstraction
    def batched(proj_data_list):
        num_batches = len(proj_data_list)
        
        collated_intrinsics = [x.intrinsics for x in proj_data_list]
        collated_extrinsics = [x.extrinsics for x in proj_data_list]
        
        batched_intrinsics = IntrinsicsUtils.batched(collated_intrinsics)
        batched_extrinsics = ExtrinsicsUtils.batched(collated_extrinsics)
        
        return ProjectionData(batched_intrinsics, batched_extrinsics, num_batches)
    
    def retrieve(proj_data, index):
        intrinsics = IntrinsicsUtils.retrieve(proj_data.instrinsics, index)
        extrinsics = ExtrinsicsUtils.retrieve(proj_data.extrinsics, index)
        return ProjectionData(intrinsics, extrinsics)