import torch
from .statistics import *

class Benchmarker:
    def __init__(self, frontend, size, config):
        self._frontend = frontend
        self._size = size

        coords_metric = MatchMetric.instance(config["coords.metric"])
        coords_criterion = MatchCriterion.instance(config["coords.criterion"], *config["coords.criterion_args"])
        features_metric = MatchMetric.instance(config["features.metric"])
        features_criterion = MatchCriterion.instance(config["features.criterion"], *config["features.criterion_args"])

        self._coords_matcher = Matcher(coords_metric, coords_criterion)
        self._features_matcher = Matcher(features_metric, features_criterion)

    def loop(self):
        count = 0

        stats = []

        while count < self._size:
            data = self._frontend.receive(blocking=True)
            (c1, c2, f1, f2) = (data.expected_kps[:, 0:2], data.actual_kps, data.expected_features, data.actual_features)

            true_dists, true_matches = self._coords_matcher.match(c1, c2)
            pred_dists, pred_matches = self._features_matcher.match(f1, f2)
            
            stats.append(Statistics(true_matches, pred_matches))            

            count += 1

        avg_accuracy = torch.tensor(list(map(lambda x: x.accuracy(), stats))).mean()
        avg_precision = torch.tensor(list(map(lambda x: x.precision(), stats))).mean()
        avg_recall = torch.tensor(list(map(lambda x: x.recall(), stats))).mean()
        avg_f1 = torch.tensor(list(map(lambda x: x.f1(), stats))).mean()
            
        print(f"Average accuracy: {avg_accuracy}")
        print(f"Average precision: {avg_precision}")
        print(f"Average recall: {avg_recall}")
        print(f"Average F1 score: {avg_f1}")