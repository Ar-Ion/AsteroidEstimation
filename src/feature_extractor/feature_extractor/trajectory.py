import numpy as np
from matplotlib import pyplot as plt

class Trajectory:
    def __init__(self, initial_coords, gt_motion_model):
        self._initial_object_coords = gt_motion_model.camera2object(initial_coords)
        self._gt_motion_model = gt_motion_model
        self._keypoints = []
        self._err = []

    def __del__(self):
        if len(self._err) > 50:

            plt.figure()
            plt.plot(np.array(self._keypoints))
            plt.plot(np.array(self._keypoints) + np.array(self._err))
            plt.show()

    def add_measured_keypoint(self, kp, dist):
        (x_meas, y_meas) = kp.pt
        (x_gt, y_gt, z_gt) = self._gt_motion_model.object2camera(self._initial_object_coords)
        (x_err, y_err) = (x_meas - x_gt, y_meas - y_gt)


        self._keypoints.append((x_meas, y_meas))

        err_vec = np.array([x_err, y_err])
        self._err.append(err_vec)
