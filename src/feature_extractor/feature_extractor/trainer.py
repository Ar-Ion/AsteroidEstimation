import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.nn import MSELoss
from torch.optim import SGD
import torch.multiprocessing as mp
import sys

class TrainContext:
    def __init__(self, backend, motion_model, device, batch_size):
        self.backend = backend
        self.motion_model = motion_model
        self.device = device
        self.batch_size = batch_size

        # Optimizers 
        self.optimizer = SGD(backend.get_model().parameters(), lr=0.01, momentum=0.9)

        # Statistics
        self.epoch = 0


class Trainer:

    # Evaluates the performance of a given backend algorithm against a ground-truth motion model
    def __init__(self, backend, motion_model):
        self._backend = backend
        self._motion_model = motion_model

        # CUDA configuration
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # KL-divergence loss
        self._kl_loss = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")

        # Batch buffer
        self._batch_extrinsics = []
        self._batch_images = []
        self._batch_depths = []
        self._batch_size = 8

        # Pipeline
        self._context = TrainContext(backend, motion_model, self._device, self._batch_size)
        self._train_thread = None

    def loss(distance_matrix, match_matrix):
        epsilon = 8
        ld = 250
        mp = 1
        mn = 0.2
        
        s = torch.zeros_like(distance_matrix)
        s[distance_matrix < epsilon] = 1
        
        hinge_loss = ld * s * torch.max(torch.tensor(0), mp - match_matrix) + (1 - s) * torch.max(torch.tensor(0), match_matrix - mn)

        tp = (s * match_matrix).sum()/torch.count_nonzero(s)
        tn = ((1 - s) * (1 - match_matrix)).sum()/torch.count_nonzero((1 - s))

        #print("True positive: " + str(tp))
        #print("True negative: " + str(tn))

        return 2 - tp - tn

        # normed_match_matrix = match_matrix**2

        # # torch.log(torch.nn.functional.normalize(1 + distance_matrix, dim=0)) +
        # expected_likelihood = torch.log_softmax(-distance_matrix**2/(2*sigma*sigma), dim=0)
        # predicted_likelihood = torch.log(normed_match_matrix / torch.sum(normed_match_matrix, dim=0))

        # kl_div = torch.sum(torch.exp(expected_likelihood) * (expected_likelihood - predicted_likelihood), dim=0)
        
        # return kl_div.mean()

    def on_input(self, image, depth):
        self._batch_extrinsics.append(self._motion_model.get_camera_extrinsics())
        self._batch_images.append(image)
        self._batch_depths.append(depth)

        if len(self._batch_images) == self._batch_size:

            # if self._train_thread != None:
            #     self._train_thread.join()
            #     self._context = self._context_queue.get()

            # self._train_thread = mp.spawn(
            #     Trainer.on_batch, 
            #     args=(
            #         self._context_queue,
            #         np.array(self._batch_extrinsics), 
            #         np.array(self._batch_images), 
            #         np.array(self._batch_depths)
            #     ),
            #     join=False
            # )

            Trainer.on_batch(self._context,
                np.array(self._batch_extrinsics), 
                np.array(self._batch_images), 
                np.array(self._batch_depths)
            )

            self._batch_extrinsics.clear()
            self._batch_images.clear()
            self._batch_depths.clear()

    def extract_world_coords(motion_model, extrinsics, depth, kps):
        motion_model.set_camera_extrinsics(extrinsics)
        depth_value = depth[kps[:, 0], kps[:, 1]]
        cam_coords = torch.hstack([kps, depth_value[:, None]])
        return motion_model.camera2object(cam_coords)

    def get_sparse_from_batch(coords, features, batch_id):
        filter = coords[:, 0] == batch_id
        return (coords[filter, 1:], features[filter])

    def on_batch(context, extrinsics, image, depth):
        # Enable training mode
        context.backend.get_model().train()
        
        ### Extract features from batch    
        kps, dess = context.backend.extract_features(image)
        
        ### Construct ground-truth feature trajectories and batch loss function
        # Send depth map to GPU
        torch_depth = torch.from_numpy(depth).to(context.device)

        (last_kps, last_dess) = Trainer.get_sparse_from_batch(kps, dess, 0)
        last_coords = Trainer.extract_world_coords(context.motion_model, extrinsics[0], torch_depth[0], last_kps)

        batch_loss = 0

        # Keep last matrices on batch scope for evaluation
        distance_matrix = None
        match_matrix = None

        for i in range(1, context.batch_size):
            # Update the motion model, so that we can compare the result of the backend with the ground-truth
            context.motion_model.set_camera_extrinsics(extrinsics[i])

            # Get coordinates and features for the current batch
            (batch_kps, batch_dess) = Trainer.get_sparse_from_batch(kps, dess, i)

            # Project the previous feature coordinates in the camera frame with the new extrinsics matrix
            expected_cam_coords = context.motion_model.object2camera(last_coords)
            non_nan_filter = ~torch.any(expected_cam_coords.isnan(), dim=0)
            expected_kps = expected_cam_coords[0:2, non_nan_filter].T
            expected_features = last_dess[non_nan_filter]

            # Compute the distance and cosine similarity matrices
            distance_matrix = torch.cdist(batch_kps.float(), expected_kps)
            match_matrix = batch_dess @ expected_features.T
        
            # Add to global batch loss function
            batch_loss += Trainer.loss(distance_matrix, match_matrix)

            # Latch the current values, so that they are available for the next iteration
            last_kps = batch_kps
            last_dess = batch_dess
            last_coords = Trainer.extract_world_coords(context.motion_model, extrinsics[i], torch_depth[i], batch_kps)

        # Perform SGD over the whole batch
        batch_loss.backward()
        context.optimizer.step()
        context.optimizer.zero_grad()

        context.epoch += 1

        print(f'Epoch: {context.epoch} Loss: {batch_loss.item() / context.batch_size}')