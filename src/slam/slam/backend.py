import time 
import platform 
import multiprocessing as mp 
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import quaternion

from ament_index_python.packages import get_package_share_directory
        
from feature_matcher.backends.criteria import MatchCriterion
from feature_matcher.backends import Matcher
from astronet_utils import ExtrinsicsUtils, IntrinsicsUtils, ProjectionUtils
from astronet_msgs import ProjectionData

from .pyslam.config import Config
from .pyslam.slam import Slam, SlamState
from .pyslam.camera  import PinholeCamera
from .pyslam.mplot_thread import Mplot2d, Mplot3d
if platform.system()  == 'Linux':     
    from .pyslam.display2D import Display2D  #  !NOTE: pygame generate troubles under macOS!
from .pyslam.viewer3D import Viewer3D
from .pyslam.utils_sys import getchar, Printer 
from .pyslam.feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from .pyslam.feature_manager import feature_manager_factory
from .pyslam.feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from .pyslam.feature_matcher import feature_matcher_factory, FeatureMatcherTypes
from .pyslam.feature_tracker_configs import FeatureTrackerConfigs
from .pyslam.parameters import Parameters  

class SLAMBackend:
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

        #matcher = Matcher.instance(config["matcher"], *matcher_args)
        #criterion = MatchCriterion.instance(config["criterion"], *criterion_args)
        #matcher.set_criterion(criterion)
                
        print("Loading GPU...")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("GPU loaded. Using compute device: " + str(self._device))
        
        pyslam_config = Config(get_package_share_directory('slam') + '/pyslam')
                
        cam = PinholeCamera(pyslam_config.cam_settings['Camera.width'], pyslam_config.cam_settings['Camera.height'],
                            pyslam_config.cam_settings['Camera.fx'], pyslam_config.cam_settings['Camera.fy'],
                            pyslam_config.cam_settings['Camera.cx'], pyslam_config.cam_settings['Camera.cy'],
                            pyslam_config.DistCoef, pyslam_config.cam_settings['Camera.fps'])
    
        
        num_features=2000 

        tracker_type = FeatureTrackerTypes.DES_BF      # descriptor-based, brute force matching with knn 
        #tracker_type = FeatureTrackerTypes.DES_FLANN  # descriptor-based, FLANN-based matching 

        # select your tracker configuration (see the file feature_tracker_configs.py) 
        # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT, CONTEXTDESC
        tracker_config = FeatureTrackerConfigs.COFFEE
        tracker_config['num_features'] = num_features
        tracker_config['tracker_type'] = tracker_type
        
        print('tracker_config: ',tracker_config)    
        feature_tracker = feature_tracker_factory(**tracker_config)
        
        # create SLAM object 
        self._slam = Slam(cam, feature_tracker, None)

    def get_projection(self, data):
        env_pose = data.env_data.pose
        cam_pose = data.robot_data.cam_data[0].pose
        
        env_quat = np.quaternion(*env_pose.rot)
        cam_quat = np.quaternion(*cam_pose.rot)

        rot = np.quaternion(np.sqrt(2)/2, -np.sqrt(2)/2, 0, 0) # I found this matrix empirically... Ideally, use cam_quat instead

        env2cam_trans = quaternion.rotate_vectors(cam_quat, env_pose.trans - cam_pose.trans)
        env2cam_quat = rot * env_quat

        intrinsics = IntrinsicsUtils.from_K(np.array(data.robot_data.cam_data[0].k))
        extrinsics = ExtrinsicsUtils.from_SE3_7D(env2cam_trans, env2cam_quat)

        return ProjectionData(intrinsics, extrinsics)

    def loop(self):
        count = 0  

        errors = []
        error_matrices = []

        prev_proj = None
        prev_pose = None
        
        try:
            while count < self._size:
                data = self._frontend.receive(blocking=True)
                camera_data = data.robot_data.cam_data[0]        
                img_id = count

                plt.imshow(camera_data.image)
                plt.pause(1.0)

                # img = np.swapaxes(camera_data.image, 0, 1)
                                
                # self._slam.track(img, img_id, count)  # main SLAM function 
                
                # pose = self._slam.tracking.f_cur.pose
                # proj = self.get_projection(data)

                # if prev_proj != None:
                #     R = pose[:3, :3] @ prev_pose[:3, :3].T

                #     extrinsics_prev = prev_proj.extrinsics.M.cpu().numpy()
                #     extrinsics_next = proj.extrinsics.M.cpu().numpy()
                
                #     ground_truth = extrinsics_next[:3, :3] @ extrinsics_prev[:3, :3].T
                #     error = R @ ground_truth.T

                #     error_vec, _ = cv2.Rodrigues(error)
                #     mag_vec, _ = cv2.Rodrigues(ground_truth)
                    
                #     errors.append(np.linalg.norm(error_vec) / np.linalg.norm(mag_vec))
                #     error_matrices.append(error)

                # count += 1

                # prev_proj = proj
                # prev_pose = pose

            np_errors = np.array(errors)
            np_error_matrices = np.array(error_matrices)
        
            plt.figure()
            plt.plot(np_errors)
            plt.show()

            bias, _ = cv2.Rodrigues(np.mean(np_error_matrices, axis=0))
            
            print(np.mean(np_errors))
            print(bias)
                
        except KeyboardInterrupt as exc:
            self._slam.quit()
            raise exc