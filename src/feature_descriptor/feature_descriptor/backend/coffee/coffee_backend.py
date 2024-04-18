import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.sparse import lil_matrix, coo_matrix
from scipy.interpolate import CubicSpline
from scipy.ndimage import label
from . import Backend
import cProfile

# Celestial Occlusion Fast FEature Extractor
class COFFEEBackend(Backend):

    def __init__(self):
        self._edge_filter = np.array([[+1, -1]])
        
    def extract_rays(self, image):
        out = convolve2d(image, self._edge_filter, 'same')
        sparse = lil_matrix(out)
        
        ray_map = np.zeros_like(image)
        
        for y, row in enumerate(sparse.rows):
            last_neg_edge = None
            last_pos_edge = None
            
            for x in row:
                
                value = sparse[y, x]
                                
                if value < 0:
                    last_neg_edge = x

                if value > 0:
                    last_pos_edge = x
                    
                if last_pos_edge is not None and last_neg_edge is not None and last_neg_edge < last_pos_edge:
                    ray_map[y][last_neg_edge] = last_pos_edge - last_neg_edge
                    last_neg_edge = None
                                        
        return ray_map
    
    def extract_shadows(self, rays):
        structuring_element = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        labels, n_features = label(rays, structure=structuring_element)
        
        sparse_rays = coo_matrix(rays)
        
        features = [[] for _ in range(n_features+1)]
        
        for i in range(sparse_rays.nnz):
            z = sparse_rays.data[i]
            y = sparse_rays.row[i]
            x = sparse_rays.col[i]

            cluster = labels[y][x]
                        
            features[cluster].append((x, y, z))
        
        filtered_features = []
        
        for feature in features:
            if len(feature) >= 12:
                filtered_features.append(feature)
        
        return filtered_features
        
    def create_keypoints(self, features):
        
        keypoints = []
        
        # The keypoints are defined as the average location of where the ray impacts the object (start of shadow)
        for feature in features:
            points = np.array(feature)
            kp = cv2.KeyPoint(np.mean(points[:, 0]), np.mean(points[:, 1]), np.mean(points[:, 2]))
            keypoints.append(kp)
            
        return keypoints
    
    def create_descriptors(self, features):
        
        # Define the dimension of the feature descriptor (empirical)
        dimension = 32
        
        descriptors = np.zeros((len(features), dimension, 3))
        z_queries = np.linspace(0, 1, dimension)
                
        for i in range(len(features)):
            points = np.array(features[i])
            
            # Get an exact number of description points
            indices = np.linspace(0, 1, len(points))
            
            # Use cubic spline interpolation
            cs = CubicSpline(indices, points)
            interp = cs(z_queries)

            # Or use linear interpolation (less performant)
            #z_interp = np.interp(z_queries, z_indices, normalized_z)

            descriptors[i] = interp
            
        return descriptors

    def normalize_descriptors(self, descriptors):
        x_avg = np.mean(descriptors[:, :, 0], axis=1)  
        y_avg = np.mean(descriptors[:, :, 1], axis=1)  
        z_avg = np.mean(descriptors[:, :, 2], axis=1) 
        x_std = np.ones((descriptors.shape[0]))#np.std(descriptors[:, :, 0], axis=1)
        y_std = np.ones((descriptors.shape[0]))#np.std(descriptors[:, :, 1], axis=1)

        centering_vec = np.array([[x_avg], [y_avg], np.zeros((1, descriptors.shape[0]))]).T
        scaling_vec = np.array([[x_std], [y_std], [1/z_avg]]).T

        centered_desc = (descriptors - centering_vec) * scaling_vec

        return centered_desc
    
    def zip_descriptors(self, descriptors):
        return descriptors.reshape(*descriptors.shape[:-2], -1)

                     
    def wrap(self, image):
        # TODO: rotate image, so that the shadows are aligned horizontally
    
        # Preprocess the data by thresholding
        image[image <= 100] = 0
        image[image > 100] = 1

        # Extract rays (horizontal cast shadow rays)
        rays = self.extract_rays(image)

        # Cluster the rays into full shadows to reduce the number of features
        shadows = self.extract_shadows(rays)
     
        keypoints = self.create_keypoints(shadows)
        descriptors = self.create_descriptors(shadows)

        normalized_desc = self.normalize_descriptors(descriptors)

        zipped_desc = self.zip_descriptors(normalized_desc)

        # reencoded = np.zeros_like(image)
        
        # nonzero_rows, nonzero_cols = rays.nonzero()

        # for row, col in zip(nonzero_rows, nonzero_cols):
        #     val = rays[row, col]
            
        #     reencoded[row][col] = 255

        #     for h in range(val-1):
        #         reencoded[row][col+h+1] = 63

        # output = cv2.drawKeypoints(reencoded, keypoints, 0, (0, 0, 255)) 
        
        # plt.figure()
        # plt.imshow(output)
        # plt.figure()
        # plt.imshow(image)
        # plt.show()
                
        return (tuple(keypoints), np.array(zipped_desc, dtype=np.float32))
                
    def extract_features(self, image):
        #cProfile.runctx('self.wrap(image)', globals(), locals())
        return self.wrap(image)

    def get_match_norm(self):
        return cv2.NORM_L2