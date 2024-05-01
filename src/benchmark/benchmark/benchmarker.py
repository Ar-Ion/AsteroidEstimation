import torch
from .statistics import *

class Benchmarker:
    def __init__(self, frontend, size, config):
        self._frontend = frontend
        self._size = size
       
        coords_criterion_args = config["coords.criterion_args"]
        features_criterion_args = config["features.criterion_args"]
        
        if not coords_criterion_args:
            coords_criterion_args = []
            
        if not features_criterion_args:
            features_criterion_args = []

        coords_metric = MatchMetric.instance(config["coords.metric"])
        coords_criterion = MatchCriterion.instance(config["coords.criterion"], *coords_criterion_args)
        features_metric = MatchMetric.instance(config["features.metric"])
        features_criterion = MatchCriterion.instance(config["features.criterion"], *features_criterion_args)

        self._coords_matcher = Matcher(coords_metric, coords_criterion)
        self._features_matcher = Matcher(features_metric, features_criterion)

        print("Loading GPU...")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("GPU loaded. Using compute device: " + str(self._device))

    def loop(self):
        count = 0

        stats = []
        feature_count = 0

        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            while count < self._size:
                data = self._frontend.receive(blocking=True)
                (c1, c2, f1, f2) = (data.expected_kps, data.actual_kps, data.expected_features, data.actual_features)

                true_dists, true_matches = self._coords_matcher.match(c1.to(self._device), c2.to(self._device))
                pred_dists, pred_matches = self._features_matcher.match(f1.to(self._device), f2.to(self._device))
                
                stats.append(Statistics(true_matches, pred_matches))            
                feature_count += 0.5*(f1.size(0) + f2.size(0))

                count += 1

                if count % 100 == 0:
                    print("Computed statistics for " + f"{count/self._size:.0%}" + " of the described data")

        avg_accuracy = torch.tensor(list(map(lambda x: x.accuracy(), stats))).mean()
        avg_precision = torch.tensor(list(map(lambda x: x.precision(), stats))).mean()
        avg_recall = torch.tensor(list(map(lambda x: x.recall(), stats))).mean()
        avg_f1 = torch.tensor(list(map(lambda x: x.f1(), stats))).mean()
        avg_feature_count = feature_count / self._size
        avg_true_count = torch.tensor(list(map(lambda x: x.true_count(), stats))).mean()
        avg_positive_count = torch.tensor(list(map(lambda x: x.positive_count(), stats))).mean()
            
        print(f"Average accuracy: {avg_accuracy}")
        print(f"Average precision: {avg_precision}")
        print(f"Average recall: {avg_recall}")
        print(f"Average F1 score: {avg_f1}")
        print(f"Average feature count: {avg_feature_count}")
        print(f"Average true count: {avg_true_count}")
        print(f"Average positive count: {avg_positive_count}")