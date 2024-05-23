from abc import ABC, abstractmethod
import importlib

class MatchCriterion(ABC):
    @abstractmethod
    def apply(self, dist_matrix):
        pass

    def instance(type, *args):
        module = importlib.import_module("feature_matcher.backends.criteria")
        criterion_class = getattr(module, type)
        criterion = criterion_class(*args)
        return criterion