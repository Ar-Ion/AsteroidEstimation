class Trajectory:
    def __init__(self, initial_coords, gt_motion_model):
        self._initial_coords = initial_coords
        self._gt_motion_model = gt_motion_model
        self._keypoints = []

    def add_measured_keypoint(self, kp, dist):
        #self._keypoints.append((kp, dist))
        (x_meas, y_meas) = kp.pt
        (x_gt, y_gt) = self._gt_motion_model.propagate(self._initial_coords)
