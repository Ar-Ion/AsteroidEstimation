class ProjectionData:
    def __init__(self, intrinsics, extrinsics, num_batches=-1):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.num_batches = num_batches
        
    class IntrinsicsData:
        def __init__(self, K, K_inv, num_batches=-1):
            self.K = K
            self.K_inv = K_inv
            self.num_batches = num_batches
        
    class ExtrinsicsData:
        def __init__(self, M, M_inv, num_batches=-1):
            self.M = M
            self.M_inv = M_inv
            self.num_batches = num_batches
        
    