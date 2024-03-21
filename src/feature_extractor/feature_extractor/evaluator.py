import cv2
import numpy as np

from feature_extractor.trajectory import Trajectory
from feature_extractor.statistics import ErrorStatistics

class Evaluator:

    # Evaluates the performance of a given backend algorithm against a ground-truth motion model
    def __init__(self, backend, motion_model):
        self._backend = backend
        self._motion_model = motion_model
        self._stats = ErrorStatistics()

         # Bruteforce matcher works better for small number of features
        self._feature_matcher = cv2.BFMatcher(backend.get_match_norm(), crossCheck=True)
        self._prev_features = (None, None)
        self._prev_trajectories = None

    # Timestamp in nanoseconds
    def on_input(self, timestamp, image, depth):
        prev_kps, prev_dess = self._prev_features
        self._prev_features = self._backend.extract_features(image)
        next_kps, next_dess = self._prev_features

        new_trajectories = [None] * len(next_kps)

        # Propagate the motion model, so that we can compare the result of the backend with the ground-truth
        self._motion_model.propagate()

        # Only run if previous keypoints were recorded
        if prev_kps != None:
            # Match features between frames
            matches = self._feature_matcher.match(prev_dess, next_dess)        
            
            # Each matching keypoint is added to the previously initialized trajectory
            for match in matches:
                next_kp = next_kps[match.trainIdx]
                new_trajectory = self._prev_trajectories[match.queryIdx]
                new_trajectory.add_measured_keypoint(next_kp, match.distance)
                new_trajectories[match.trainIdx] = new_trajectory

        # If a keypoint does not have a matching feature in the previous frame, initialize a new trajectory
        for i in range(len(new_trajectories)):
            if new_trajectories[i] == None:

                unknown_kp = next_kps[i]

                # Extract x-y-z pseudo-coordinates in camera frame (ground truth)
                x = unknown_kp.pt[0]
                y = unknown_kp.pt[1]
                z = depth[int(x)][int(y)]

                new_trajectories[i] = Trajectory(
                    np.array([[x], [y], [z]]), 
                    self._motion_model, 
                    trajectory_complete_cb=self._stats.add_trajectory
                )

        self._prev_trajectories = new_trajectories