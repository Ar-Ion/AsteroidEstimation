import os
import torch
import pandas as pd
from matplotlib import pyplot as plt

from feature_matcher.backends.criteria import MatchCriterion
from feature_matcher.backends.metrics import MatchMetric
from feature_matcher.backends import Matcher
from astronet_utils import MotionUtils, PointsUtils, ProjectionUtils
from astronet_frontends import DriveClientFrontend

from .statistics import Statistics

class PrecisionRecall:
    def __init__(self, frontend, size, config, output_params):
        self._frontend = frontend
        self._size = size
       
        keypoints_criterion_args = config["keypoints.criterion_args"]
        features_criterion_args = config["features.criterion_args"]
        matcher_args = config["features.matcher_args"]
        plots_dir = os.path.join(output_params["path"], "precision_recall")
    
        if not keypoints_criterion_args:
            keypoints_criterion_args = []
            
        if not features_criterion_args:
            features_criterion_args = []
            
        if not matcher_args:
            matcher_args = []
            
        os.makedirs(plots_dir, exist_ok=True)

        self._f1_csv_out = os.path.join(plots_dir, "f1.csv")
        self._pr_csv_out = os.path.join(plots_dir, "pr.csv")
        self._roc_csv_out = os.path.join(plots_dir, "roc.csv")

        self._keypoints_metric = MatchMetric.instance(config["keypoints.metric"])
        self._keypoints_criterion = MatchCriterion.instance(config["keypoints.criterion"], *keypoints_criterion_args)
        self._matcher = Matcher.instance(config["features.matcher"], *matcher_args)
        self._matcher_criterion = config["features.criterion"]
        self._matcher_criterion_args = config["features.criterion_args"]

        print("Loading GPU...")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("GPU loaded. Using compute device: " + str(self._device))

    def loop(self):                
        baseline = torch.log10(torch.tensor(self._matcher_criterion_args[0]))
        #param_sweep = torch.logspace(baseline - 0.5, baseline + 0.3, 100)
        #param_sweep = torch.logspace(baseline - 3, 0, 100)
        param_sweep = torch.linspace(0.5, 1, 100)
        
        precision_recall = []
        f1 = []
        roc = []
        
        for param_idx in range(len(param_sweep)):
            
            param = param_sweep[param_idx]
            
            stats = []
            count = 0

            while count < self._size:
                input = self._frontend.receive(blocking=True)
                
                chunks = MotionUtils.create_chunks(input, 1024, 1024)                        
                valid_chunks = list(filter(MotionUtils.is_valid, chunks))
                valid_batch = MotionUtils.batched(valid_chunks)
                valid_batch_gpu = MotionUtils.to(valid_batch, device=self._device)
                                    
                for idx in range(valid_batch_gpu.num_batches):
                    data = MotionUtils.retrieve(valid_batch_gpu, idx)
                
                    prev_points_25D = torch.hstack((data.prev_points.kps, data.prev_points.depths[:, None]))
                    world_points_3D = ProjectionUtils.camera2object(data.prev_points.proj, prev_points_25D)
                    reproj_points_25D = ProjectionUtils.object2camera(data.next_points.proj, world_points_3D)
                    reproj_points_2D = reproj_points_25D[:, 0:2]

                    true_dists = self._keypoints_metric.dist(reproj_points_2D, data.next_points.kps)
                    true_matches = self._keypoints_criterion.apply(true_dists)
                    
                    features_criterion = MatchCriterion.instance(self._matcher_criterion, param)
                    self._matcher.set_criterion(features_criterion)
                    pred_dists, pred_matches = self._matcher.match(data)
                    
                    stats.append(Statistics(true_dists, true_matches, pred_matches))            
                
                count += 1

                if count % 10 == 0:
                    print("Computed statistics for " + f"{count/self._size:.0%}" + " of the described data")

            print("Precision-recall sweep status: " + f"{param_idx/len(param_sweep):.0%}")

            avg_precision = torch.tensor(list(map(lambda x: x.precision(), stats))).nanmean()
            avg_recall = torch.tensor(list(map(lambda x: x.recall(), stats))).nanmean()
            avg_f1 = torch.tensor(list(map(lambda x: x.f1(), stats))).nanmean()
            avg_fpr = torch.tensor(list(map(lambda x: x.fpr(), stats))).nanmean()

            if 0 <= avg_precision and avg_precision <= 1:
                precision_recall.append(torch.tensor((avg_recall, avg_precision)))

            if 0 <= avg_f1 and avg_f1 <= 1:
                f1.append(torch.tensor((param, avg_f1)))
                
            if 0 <= avg_fpr and avg_fpr <= 1:
                roc.append(torch.tensor((avg_fpr, avg_recall)))
            
        precision_recall.append(torch.tensor((0, 1)))
        precision_recall.append(torch.tensor((1, 0)))
        roc.append(torch.tensor((0, 0)))
        roc.append(torch.tensor((1, 1)))
        
        precision_recall.sort(key=lambda x: x[0])
        f1.sort(key=lambda x: x[0])
        roc.sort(key=lambda x: x[0])
                        
        precision_recall_curve = torch.vstack(precision_recall).cpu()
        f1_curve = torch.vstack(f1).cpu()
        roc_curve = torch.vstack(roc).cpu()
        
        print(f"Best F1 score: {torch.max(f1_curve[:, 1])}")
        print(f"PR AUC score: {torch.trapz(precision_recall_curve[:, 1], precision_recall_curve[:, 0])}")
        print(f"ROC AUC score: {torch.trapz(roc_curve[:, 1], roc_curve[:, 0])}")
        
        df = pd.DataFrame(precision_recall_curve)
        df.to_csv(self._pr_csv_out)
        
        df = pd.DataFrame(f1_curve)
        df.to_csv(self._f1_csv_out)
        
        df = pd.DataFrame(roc_curve)
        df.to_csv(self._roc_csv_out)
                
        plt.figure(figsize=(8, 6))
        plt.plot(precision_recall_curve[:, 0], precision_recall_curve[:, 1], marker='o', linestyle='-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(8, 6))
        plt.plot(roc_curve[:, 0], roc_curve[:, 1], marker='o', linestyle='-')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(8, 6))
        plt.plot(f1_curve[:, 0], f1_curve[:, 1], marker='o', linestyle='-')
        plt.xlabel('Parameter')
        plt.ylabel('F1-score')
        plt.title('F1-score parameter sweep')
        plt.grid(True)
        plt.ylim([0.0, 1.0])
        plt.show()