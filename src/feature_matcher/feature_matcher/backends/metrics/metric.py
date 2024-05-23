from abc import ABC, abstractmethod
import importlib

class MatchMetric(ABC):
    @abstractmethod
    def dist(a, b):
        pass

    def instance(type):
        module = importlib.import_module("feature_matcher.backends.metrics")
        metric_class = getattr(module, type)
        return metric_class