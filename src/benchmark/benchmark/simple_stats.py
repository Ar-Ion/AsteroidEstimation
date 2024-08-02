import torch
from matplotlib import pyplot as plt

from feature_matcher.backends.criteria import MatchCriterion, GreaterThan, LessThan, Intersection, MinRatio
from feature_matcher.backends.metrics import MatchMetric
from feature_matcher.backends import Matcher
from astronet_utils import MotionUtils, PointsUtils, ProjectionUtils

from .statistics import Statistics

class SimpleStats:
    def __init__(self, frontend, size, config):
        self._frontend = frontend
        self._size = size
       
        keypoints_criterion_args = config["keypoints.criterion_args"]
        features_criterion_args = config["features.criterion_args"]
        matcher_args = config["features.matcher_args"]
        
        if not keypoints_criterion_args:
            keypoints_criterion_args = []
            
        if not features_criterion_args:
            features_criterion_args = []
            
        if not matcher_args:
            matcher_args = []

        self._keypoints_metric = MatchMetric.instance(config["keypoints.metric"])
        self._keypoints_criterion = Intersection(MinRatio(1.0), LessThan(8))#MatchCriterion.instance(config["keypoints.criterion"], *keypoints_criterion_args)
        
        self._matcher = Matcher.instance(config["features.matcher"], *matcher_args)
        features_criterion = MatchCriterion.instance(config["features.criterion"], *features_criterion_args)
        self._matcher.set_criterion(features_criterion)

        print("Loading GPU...")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("GPU loaded. Using compute device: " + str(self._device))

    def loop(self):
        count = 0

        stats = []
        bias = []
        feature_count = 0
        
        avg_batch_size = 0

        while count < self._size:
            input = self._frontend.receive(blocking=True)
                        
            # chunks = MotionUtils.create_chunks(input, 1024, 1024)

            # valid_chunks = list(filter(MotionUtils.is_valid, chunks))
            
            batch = MotionUtils.batched([input])
            batch_gpu = MotionUtils.to(batch, device=self._device)

            avg_batch_size += batch_gpu.num_batches
                
            for idx in range(batch_gpu.num_batches):
                data = MotionUtils.retrieve(batch_gpu, idx)
                
                prev_points_25D = torch.hstack((data.prev_points.kps, data.prev_points.depths[:, None]))
                world_points_3D = ProjectionUtils.camera2object(data.prev_points.proj, prev_points_25D)
                reproj_points_25D = ProjectionUtils.object2camera(data.next_points.proj, world_points_3D)
                reproj_points_2D = reproj_points_25D[:, 0:2]

                true_dists = self._keypoints_metric.dist(reproj_points_2D, data.next_points.kps)
                true_matches = self._keypoints_criterion.apply(true_dists)
                                     
                distance_matrix = (reproj_points_2D[:, None, :] - data.next_points.kps[None, :, :]).squeeze()[:, :, 1]
                bias.append((distance_matrix[true_matches.to(dtype=torch.bool)]))
                
                
                
                pred_dists, pred_matches = self._matcher.match(data)
                                    
                indices = torch.nonzero(pred_matches)
        
                all_coords_prev = data.prev_points.kps
                all_coords_next = data.next_points.kps
                                
                coords_prev = all_coords_prev[indices[:, 0]]
                coords_next = all_coords_next[indices[:, 1]]
                
                # print(coords_prev - coords_next)

                # plt.figure()
                # hist = (coords_prev - coords_next).to(dtype=torch.float).norm(p=1, dim=1).cpu()
                # plt.hist(hist[None, :], bins=50)
                # plt.show()
                
                local_stats = Statistics(true_dists, true_matches, pred_matches)
                local_feature_count = 0.5*(data.prev_points.features.size(0) + data.next_points.features.size(0))

                stats.append(local_stats)  
                feature_count += local_feature_count
                
            count += 1

            if count % 100 == 0:
                print("Computed statistics for " + f"{count/self._size:.0%}" + " of the described data")

        # plt.figure()
        # plt.hist(torch.hstack(bias).cpu().numpy(), bins=200)
        # plt.show()

        avg_bias = torch.hstack(bias).mean()
                
        avg_batch_size /= self._size
        avg_accuracy = torch.tensor(list(map(lambda x: x.accuracy(), stats))).nanmean()
        avg_precision = torch.tensor(list(map(lambda x: x.precision(), stats))).nanmean()
        avg_recall = torch.tensor(list(map(lambda x: x.recall(), stats))).nanmean()
        avg_f1 = torch.tensor(list(map(lambda x: x.f1(), stats))).nanmean()
        avg_feature_count = feature_count / self._size
        avg_true_count = torch.tensor(list(map(lambda x: x.true_count(), stats))).nanmean()*avg_batch_size
        avg_positive_count = torch.tensor(list(map(lambda x: x.positive_count(), stats))).nanmean()*avg_batch_size
        avg_pixel_error = torch.tensor(list(map(lambda x: x.pixel_error(), stats))).nanmean()

        print(f"Average accuracy: {avg_accuracy}")
        print(f"Average precision: {avg_precision}")
        print(f"Average recall: {avg_recall}")
        print(f"Average F1 score: {avg_f1}")
        print(f"Average feature count: {avg_feature_count}")
        print(f"Average true count: {avg_true_count}")
        print(f"Average positive count: {avg_positive_count}")
        print(f"Average pixel location error: {avg_pixel_error}")
        print(f"Bias: {avg_bias}")