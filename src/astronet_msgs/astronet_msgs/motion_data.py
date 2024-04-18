class MotionData:
    def __init__(self, expected_kps, actual_kps, expected_features, actual_features):
        self.expected_kps = expected_kps
        self.actual_kps = actual_kps
        self.expected_features = expected_features
        self.actual_features = actual_features