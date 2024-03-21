import numpy as np
from matplotlib import pyplot as plt

class Trajectory:
    def __init__(self, initial_coords, gt_motion_model, trajectory_complete_cb=lambda kp, err: None):
        self._initial_object_coords = gt_motion_model.camera2object(initial_coords)
        self._gt_motion_model = gt_motion_model
        self._trajectory_complete_cb = trajectory_complete_cb
        self._keypoints = []
        self._errors = []
        
    def __del__(self):
        if len(self._keypoints) > 0:
            np_kps = np.array(self._keypoints, dtype=float)
            np_errs = np.array(self._errors, dtype=float)
                        
            if np.isfinite(np_kps).all() and np.isfinite(np_errs).all():
                self._trajectory_complete_cb(np_kps, np_errs)

    def add_measured_keypoint(self, kp, dist):
        (x_meas, y_meas) = kp.pt
        (x_gt, y_gt, z_gt) = self._gt_motion_model.object2camera(self._initial_object_coords)
        (x_err, y_err) = (x_gt - x_meas, y_gt - y_meas)

        meas_vec = np.array([x_meas, y_meas])
        self._keypoints.append(meas_vec)

        err_vec = np.array([x_err, y_err]).squeeze()
        self._errors.append(err_vec)
