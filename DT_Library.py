import pandas as pd
import numpy as np
from enum import Enum
from math import log2


class Node():
    def __init__(self, feature=None, children=None, TODO="Other information about that node you wana show and need."):
        # TODO: Initialize node attributes
        self.feature = feature
        self.children = children
        pass


class DecisionTree():
    def __init__(self, mode="gain", max_Depth=float("inf"), min_Samples=2,
                 pruning_threshold="What is diffrent when change mode?", TODO="Any desired hyperparameters"):
       # You can change all class properties for better performance!
       # TODO: Initialize tree hyperparameters and root node
        self.mode = mode
        self.max_Depth = max_Depth
        self.min_Samples = min_Samples
        self.pruning_threshold = pruning_threshold
        self.root = None

    # Y is the label column and X is the dataframe of the attributes
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


    # X is dataframe of features,   Y is series of labels.
    def _get_best_Feature(self, X, Y):

        # Ensuring that X is a dataframe and Y is a serie
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(Y, np.ndarray):
            Y = pd.Series(Y)

        if self.mode == "gain":
           # TODO: Calculate information gain for each feature
            max_val = float('-inf')
            best_feature = ""
            for feature in list(X.columns):
                info_gain = self._information_Gain(feature, X, Y)
                if info_gain > max_val:
                    max_val = info_gain
                    best_feature = feature

        elif self.mode == "gini":
           # TODO: Calculate gini split for each feature
            max_val = float('-inf')
            best_feature = ""
            for feature in list(X.columns):
                info_gain = self._gini_Split(feature, X, Y)
                if info_gain > max_val:
                    max_val = info_gain
                    best_feature = feature

        # TODO: Select and return best feature as Node
        # for now i put list of names of the children features no a node if required:
        bestfeature_serie = X[best_feature]
        merged_with_label = pd.concat([bestfeature_serie, Y], axis=1)
        children_list = []
        for uniqueval in merged_with_label[best_feature].unique().tolist():
            children_list.append(uniqueval)
        return Node(feature=best_feature, children=children_list)

    # notes for the infogain and ginisplit methods:
        # Some potential improvements:
        # passing dataset instead of X and Y.

    # infogain is defined for a feature.
    def _information_Gain(self, feature, X, Y): #X is the dataframe with all the feature columns.    #Y is a series of the label 
        # Make sure Y is a Series and has a usable name
        if not isinstance(Y, pd.Series):
            Y = pd.Series(Y)

        # in our dataset it is not needed becuase we have the name for label column:
        if Y.name is None:
            Y = Y.rename("target") 
       
        def entropy(y):   #y is a series for a label in dataset.
           # TODO: Calculate entropy for target variable
            #again this one is needed for our dataset because the unique values are satisfied and not satisfied:
            y = y.map({"satisfied": 1, "dissatisfied": 0 ,1: 1, 0: 0})
            yes_counts = (y == 1).sum()
            total_counts = y.count()
            fraction = yes_counts / total_counts
            if fraction == 0 or fraction == 1:
                return 0
            result = -fraction * log2(fraction) - (1 - fraction) * log2(1 - fraction)
            return result

        desired_column = X[feature]  #this is a series.
        merged = pd.concat([desired_column, Y], axis=1) #this is a dataframe of 2 columns:one is the dataset's possible answers for the feature selected.
                                                           #the other is the label corresponding to each sample of our dataset.  
        
        # calculating the sum of the entropies for each sub_dataframes filtered by different answers for that attribute:
        subentropies_sum = 0
        for value in desired_column.unique():  #returns an array of the unique values that appear in that Series
            # TODO: Calculate weighted entropy for each value
            filtered_df = merged[merged[feature] == value] #filtering by row
            weight = len(filtered_df) / len(merged)
            subentropies_sum += weight * entropy(filtered_df[Y.name].copy())
        
        # TODO: Return information gain
        return entropy(Y.copy()) - subentropies_sum
    

    def _gini_Split(self, feature, X, Y): #feature is just the name of feature.

        # Make sure Y is a Series and has a usable name
        if not isinstance(Y, pd.Series):
            Y = pd.Series(Y)     

        # keyvalue error handling:
        if Y.name is None:
            Y = Y.rename("target")

        def gini(y):  #y is a series of label given as an input
           # TODO: Calculate gini impurity for target variable
            #1- sigma(p(i)**2)
            y = y.map({"satisfied": 1, "dissatisfied": 0 ,1: 1, 0: 0})
            total_count = len(y)
            yes_counts = (y == 1).sum()
            fraction = yes_counts / total_count
            if fraction == 0 or fraction == 1:
                return 0
            result = 1 - (fraction ** 2 + (1 - fraction) ** 2)
            return result

        desired_column = X[feature]
        merged = pd.concat([desired_column, Y], axis=1)
        
        # Sum of the weighted gini's for sub_dataframes which is a rowly filtered version of main dataframe given as input
        #  based on the unique answers of the feature given again as the input :
        Sub_gini_sum = 0
        for value in desired_column.unique():
            # TODO: Calculate weighted gini for each value
            filtered_df = merged[merged[feature] == value]
            weight = len(filtered_df) / len(merged)
            Sub_gini_sum += weight * gini(filtered_df[Y.name].copy()) #passing a filtered series to the gini method.
            

        # TODO: Return gini split value
        return Sub_gini_sum

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
