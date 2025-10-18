import pandas as pd
import numpy as np

class Node():
    def __init__(self, feature=None, children=None, TODO="Other information about that node you wana show and need."):
        # TODO: Initialize node attributes
        pass

class DecisionTree():
    def __init__(self, mode="gain or gini?", max_Depth=float("inf"), min_Samples=2,
                 pruning_threshold="What is diffrent when change mode?", TODO="Any desired hyperparameters"):
       # You can change all class properties for better performance!
       # TODO: Initialize tree hyperparameters and root node
       pass

    def _create_Tree(self, X, Y, depth=0):
        num_Samples = len(Y)
        
        # Check stopping conditions (Pre-Pruning)
        if num_Samples >= self.min_Samples and depth < self.max_Depth:
            best_Feature = self._get_best_Feature(X, Y)
            children = []
            # Check gain or gini!
            for (Xi, Yi) in best_Feature["feature_values"]:
                # TODO: Recursively create child nodes
                pass
            
        # TODO: Create leaf node with predicted value
        return Node()

    def _get_best_Feature(self, X, Y):
        if self.mode == "gain":
           # TODO: Calculate information gain for each feature
           pass

        elif self.mode == "gini":
           # TODO: Calculate gini split for each feature
           pass
        
        # TODO: Select and return best feature as Node
        pass
        
    def _information_Gain(self, feature, X, Y):
        
        def entropy(y):
           # TODO: Calculate entropy for target variable
           pass
      
        for value in feature.unique():
            # TODO: Calculate weighted entropy for each value
            pass
        
        # TODO: Return information gain
        pass
    
    def _gini_Split(self, feature, X, Y):
        
        def gini(y):
           # TODO: Calculate gini impurity for target variable
           pass

        for value in feature.unique():
            # TODO: Calculate weighted gini for each value
            pass

        # TODO: Return gini split value
        pass

    def _calculate_Value(self, Y):
        # Where is it used and what does it do?
        return max(set(Y), key=list(Y).count)

    def fit(self, X_Train, Y_Train):
        self.root = self._create_Tree(X_Train, Y_Train)

    def predict(self, X):
        def _move_Tree(sample, root):
            # TODO: If leaf node return pred value, Or find recursively leaf node
            pass

        # TODO: Apply _move_Tree to each sample in X
        pass