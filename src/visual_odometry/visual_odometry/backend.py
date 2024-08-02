import torch
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

from feature_matcher.backends.criteria import MatchCriterion, Intersection, MinRatio, MaxRatio
from feature_matcher.backends import Matcher
from astronet_utils import MotionUtils
from .vo import VisualOdometry

class VOBackend:
    def __init__(self, frontend, size, config):
        self._frontend = frontend
        self._size = size
       
        matcher_args = config["matcher_args"]
        criterion_args = config["criterion_args"]

        if not matcher_args:
            matcher_args = []
            
        if not criterion_args:
            criterion_args = []
            
        if not matcher_args:
            matcher_args = []

        matcher = Matcher.instance(config["matcher"], *matcher_args)
        
        if config["criterion"] == "MinK":
            criterion = Intersection(MinRatio(1), MatchCriterion.instance(config["criterion"], *criterion_args))
        else:
            criterion = Intersection(MaxRatio(1), MatchCriterion.instance(config["criterion"], *criterion_args))

        matcher.set_criterion(criterion)
        
        self._vo = VisualOdometry(matcher)
        
        print("Loading GPU...")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("GPU loaded. Using compute device: " + str(self._device))

    def reject_outliers(self, data, m=2):
        return data[np.abs(data - np.mean(data)) < m * np.std(data)]
    
    def loop(self):
        count = 0
        
        errors_per_frame = []
        errors_per_sample = []
        
        time_stats = []

        while count < self._size:
            data = self._frontend.receive(blocking=True)
            
            data_gpu = MotionUtils.to(data, device=self._device)

            start = time.time()
            frame_out = self._vo.compute_pose(data_gpu)
            time_stats.append(time.time() - start)
            
            if frame_out != None:
                R, t = frame_out
                
                extrinsics_prev = data.prev_points.proj.extrinsics.M.cpu().numpy()
                extrinsics_next = data.next_points.proj.extrinsics.M.cpu().numpy()
            
                ground_truth = extrinsics_next[:3, :3] @ extrinsics_prev[:3, :3].T
                error = R @ ground_truth.T
                err, _ = cv2.Rodrigues(error)
                
                errors_per_frame.append(np.linalg.norm(err))
                

            samples_out = self._vo.get_sampled_poses(data_gpu, num_samples=1000)
             
            if samples_out != None:
                R, t = samples_out

                extrinsics_prev = data.prev_points.proj.extrinsics.M.cpu().numpy()
                extrinsics_next = data.next_points.proj.extrinsics.M.cpu().numpy()
            
                ground_truth = extrinsics_next[:3, :3] @ extrinsics_prev[:3, :3].T
                
                #print(R)
                #print(ground_truth)
                
                error = R @ ground_truth.T
                #error = error[None, :]
                
                origin = np.array((0, 0, 1))
                expected_translation = (origin - R @ origin)[:, 0:2]
                expected_translation = expected_translation / np.linalg.norm(expected_translation, axis=-1)[:, None]
                coherence = expected_translation[:, None] @ t[:, 0:2]
                
                print(np.mean(coherence))
                
                for i in range(len(error)):
                    #if error[i, 0, 0] > 0.9 and error[i, 1, 2] < 0:   
                    #print(f"expected {expected_translation[i]}")
                    #print(f"got {t[i]}")
                    
                    if coherence[i] > 0.0:
                        err, _ = cv2.Rodrigues(error[i])
                        
                        #sds = np.rad2deg(np.arccos((np.trace(error[i]) - 1) / 2))
                        #errors.append(sds)
                        errors_per_sample.append(np.linalg.norm(err) - np.pi)
                        #errors.append(-err)
                
            if count % 100 == 0:
                print("Analyzed " + f"{count/self._size:.0%}" + " of synthetic feature data")
                
            # np_errors = np.array(errors)

            # plt.figure()
            # plt.hist(np_errors, bins=1000)
            # plt.show()
                 
            count += 1
            
        np_errors_per_frame = np.array(errors_per_frame)
        # plt.figure()
        # plt.plot(np_errors_per_frame)
        # plt.show()

        np_errors_per_sample = np.array(errors_per_sample)
        # plt.figure()
        # plt.hist(np_errors_per_sample, bins=1000)
        # plt.show()


        bias = np.mean(np_errors_per_frame)
        std = np.std(np_errors_per_frame)
        avg_time = np.median(time_stats)
        
        print(f"Bias: {bias}")
        print(f"Std: {std}")
        print(f"Median estimation time: {avg_time}ms")
