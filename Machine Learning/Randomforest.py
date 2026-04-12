import numpy as np
import pandas as pd

class  TreeNode(object):
    def __init__(self, feature, value):
        self.feature = feature
        self.left= None
        self.right = None
        self.value = value

class RandomForest:
    def __init__(self, n_estimators, max_depth):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
    def fit(self, x, y):
