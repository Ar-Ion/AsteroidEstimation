import torch
import random
from .statistics import Statistics
from feature_matcher.backends.criteria import MatchCriterion, GreaterThan
from feature_matcher.backends.metrics import MatchMetric
from feature_matcher.backends import Matcher
from astronet_msgs import BatchedMotionData

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
        self._keypoints_criterion = MatchCriterion.instance(config["keypoints.criterion"], *keypoints_criterion_args)
        
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
        feature_count = 0
        
        avg_batch_size = 0

        while count < self._size:
            input = self._frontend.receive(blocking=True)
            chunks = input.to(self._device).create_chunks(1024, 1024)
                        
            valid_chunks = list(filter(lambda x: x.is_valid(), chunks))
                        
            batch = BatchedMotionData.from_list(valid_chunks).to(self._device)
            avg_batch_size += batch.num_batches
                
            for idx in range(batch.num_batches):
                data = batch.retrieve(idx)

                true_dists = self._keypoints_metric.dist(data.proj_kps, data.next_kps)
                true_matches = self._keypoints_criterion.apply(true_dists)
                
                pred_dists, pred_matches = self._matcher.match(data)
                            
                stats.append(Statistics(true_dists, true_matches, pred_matches))            
                feature_count += 0.5*(data.prev_features.size(0) + data.next_features.size(0))
                
            count += 1

            if count % 100 == 0:
                print("Computed statistics for " + f"{count/self._size:.0%}" + " of the described data")

        avg_batch_size /= self._size
        avg_accuracy = torch.tensor(list(map(lambda x: x.accuracy(), stats))).mean()
        avg_precision = torch.tensor(list(map(lambda x: x.precision(), stats))).mean()
        avg_recall = torch.tensor(list(map(lambda x: x.recall(), stats))).mean()
        avg_f1 = torch.tensor(list(map(lambda x: x.f1(), stats))).mean()
        avg_feature_count = feature_count / self._size
        avg_true_count = torch.tensor(list(map(lambda x: x.true_count(), stats))).mean()*avg_batch_size
        avg_positive_count = torch.tensor(list(map(lambda x: x.positive_count(), stats))).mean()*avg_batch_size
        avg_pixel_error = torch.tensor(list(map(lambda x: x.pixel_error(), stats))).mean()

        print(f"Average accuracy: {avg_accuracy}")
        print(f"Average precision: {avg_precision}")
        print(f"Average recall: {avg_recall}")
        print(f"Average F1 score: {avg_f1}")
        print(f"Average feature count: {avg_feature_count}")
        print(f"Average true count: {avg_true_count}")
        print(f"Average positive count: {avg_positive_count}")
        print(f"Average pixel location error: {avg_pixel_error}")