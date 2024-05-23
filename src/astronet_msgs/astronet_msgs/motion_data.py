class MotionData:
    def __init__(self, prev_points, next_points, num_batches=-1):
        self.prev_points = prev_points
        self.next_points = next_points
        self.num_batches = num_batches
        
    class PointsData:
        def __init__(self, kps, depths, features, proj, num_batches=-1):
            self.kps = kps
            self.depths = depths
            self.features = features
            self.proj = proj
            self.num_batches = num_batches