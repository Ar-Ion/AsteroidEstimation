import torch
from .statistics import *
from astronet_msgs import BatchedMotionData

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

        while count < self._size:
            data = self._frontend.receive(blocking=True)
            batch = BatchedMotionData.from_list([data]).to(self._device)

            true_dists, true_matches = self._coords_matcher.match(batch.proj_kps, batch.next_kps)
            pred_dists, pred_matches = self._features_matcher.match(batch.prev_features, batch.next_features)
            
            self.loss(true_dists, true_matches, pred_dists, pred_matches)
            stats.append(Statistics(true_dists, true_matches, pred_dists, pred_matches))            
            feature_count += 0.5*(batch.prev_features.size(0) + batch.next_features.size(0))
            
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
        avg_pixel_error = torch.tensor(list(map(lambda x: x.pixel_error(), stats))).mean()

        print(f"Average accuracy: {avg_accuracy}")
        print(f"Average precision: {avg_precision}")
        print(f"Average recall: {avg_recall}")
        print(f"Average F1 score: {avg_f1}")
        print(f"Average feature count: {avg_feature_count}")
        print(f"Average true count: {avg_true_count}")
        print(f"Average positive count: {avg_positive_count}")
        print(f"Average pixel location error: {avg_pixel_error}")

    def loss(self, true_dists, true_matches, pred_dists, pred_matches):      
        
        # Add dustbin to allow unmatched feature
        
        #pred_dists = torch.nn.functional.normalize(pred_dists, dim=0)
        
        weight = 1 # This parameter has little effect
        margin = 0.05
        
        # Ground truth
        matches = true_matches
        diffs = 1-true_matches

        # Hinge loss
        fn_loss = matches * torch.maximum(torch.tensor(0), 1 - pred_dists)
        fp_loss = diffs * torch.maximum(torch.tensor(0), pred_dists - margin)
        
        # Average
        fn_loss_avg = fn_loss.sum()
        fp_loss_avg = fp_loss.sum()

        print(fn_loss_avg)
        print(fp_loss_avg)
        
        # Combine
        hinge_loss = weight*fn_loss_avg + fp_loss_avg
        
        return hinge_loss.sum()