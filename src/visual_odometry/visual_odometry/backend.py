import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from feature_matcher.backends.criteria import MatchCriterion
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
        criterion = MatchCriterion.instance(config["criterion"], *criterion_args)
        matcher.set_criterion(criterion)
        
        self._vo = VisualOdometry(matcher)
        
        print("Loading GPU...")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("GPU loaded. Using compute device: " + str(self._device))

    def loop(self):
        count = 0
        
        errors = []

        while count < self._size:
            data = self._frontend.receive(blocking=True)
            
            data_gpu = MotionUtils.to(data, device=self._device)

            #out = self._vo.get_sampled_poses(data_gpu, num_samples=10)
            out = self._vo.compute_pose(data_gpu)
            
            if out != None:
                R, t = out

                extrinsics_prev = data.prev_points.proj.extrinsics.M.cpu().numpy()
                extrinsics_next = data.next_points.proj.extrinsics.M.cpu().numpy()
            
                ground_truth = extrinsics_next[:3, :3] @ extrinsics_prev[:3, :3].T
                
                error = R @ ground_truth.T

                #error = error[None, :]

                for i in range(len(error)):   
                    error_vec, _ = cv2.Rodrigues(error[i])
                    mag_vec, _ = cv2.Rodrigues(ground_truth)
                    
                    #print(error_vec)
                    #print(mag_vec)
                        
                    #sds = np.rad2deg(np.arccos((np.trace(error[i]) - 1) / 2))
                    #errors.append(sds)
                    errors.append(np.linalg.norm(error_vec) / np.linalg.norm(mag_vec))
                
            if count % 100 == 0:
                print("Analyzed " + f"{count/self._size:.0%}" + " of synthetic feature data")
                
            # np_errors = np.array(errors)

            # plt.figure()
            # plt.hist(np_errors, bins=1000)
            # plt.show()
                 
            count += 1
            
        np_errors = np.array(errors)
        
        plt.figure()
        plt.hist(np_errors[np.abs(np_errors) < 10], bins=100)
        plt.show()
        
        print(np.mean(np_errors))

    # def loop(self):
    #     count = 0
        
    #     errors = []

    #     while count < self._size:
    #         data = self._frontend.receive(blocking=True)
    #         data_gpu = MotionUtils.to(data, device=self._device)
            
    #         R, t = self._vo.compute_pose(data_gpu)
            
    #         extrinsics_prev = data.prev_points.proj.extrinsics.M.cpu().numpy()
    #         extrinsics_next = data.next_points.proj.extrinsics.M.cpu().numpy()
        
    #         ground_truth = extrinsics_next[:3, :3] @ extrinsics_prev[:3, :3].T
    #         error = R @ ground_truth.T
                        
    #         error_vec, _ = cv2.Rodrigues(error)
    #         mag_vec, _ = cv2.Rodrigues(ground_truth)
            
    #         errors.append(np.linalg.norm(error_vec) / np.linalg.norm(mag_vec))
                 
    #         count += 1
            
    #     np_errors = np.array(errors)
        
    #     plt.figure()
    #     plt.plot(np_errors)
    #     plt.show()
        
    #     print(np.mean(np_errors))