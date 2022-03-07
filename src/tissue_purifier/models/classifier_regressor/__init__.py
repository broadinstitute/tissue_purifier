# Decide what to expose
from .scikit_learn_interface import PlRegressor as MlpRegressor
from .scikit_learn_interface import PlClassifier as MlpClassifier

__all__ = ["MlpRegressor", "MlpClassifier"]
