import pandas as pd
import numpy as np
from enum import Enum
from math import log2


class alternative(Enum):
    GAIN = "gain"
    GINI = "gini"

class Node():
    def __init__(self, feature=None, children=None, TODO="Other information about that node you wana show and need."):
        # TODO: Initialize node attributes
        self.feature = feature
        self.children = children
        pass



class DecisionTree():
    def __init__(self, mode:alternative, max_Depth=float("inf"), min_Samples=2,
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

    def _get_best_Feature(self, X, Y):  #X is dataframe of features,   Y is series of labels.
        if self.mode == "gain":
           # TODO: Calculate information gain for each feature
           pass

        elif self.mode == "gini":
           # TODO: Calculate gini split for each feature
           pass
        
        # TODO: Select and return best feature as Node
        pass




    # Some potential improvements:
    # passing dataset instead of X and Y.   

    # infogain is defined for a feature.
    def _information_Gain(self, feature, X, Y): #X is the dataframe with all the feature columns.    #Y is a series of the label 
        def entropy(y):   #y is a series for a label in dataset.
           # TODO: Calculate entropy for target variable

            yes_counts = (y == "satisfied").sum()
            total_counts = y.count()

            fraction = yes_counts/total_counts

            result = -fraction*log2(fraction) - (1-fraction)*log2(1-fraction)
            return result

        desired_column = X[feature]  #this is a series.
        merged = pd.concat([desired_column, Y], axis=1) #this is a dataframe of 2 columns:one is the dataset's possible answers for the feature selected.
                                                           #the other is the label corresponding to each sample of our dataset.  
        
        # calculating the sum of the entropies for each sub_dataframes filtered by different answers for that attribute:
        subentropies_sum=0
        for value in desired_column.unique():  #returns an array of the unique values that appear in that Series
            # TODO: Calculate weighted entropy for each value
            filtered_df = merged[merged[feature]==value] #filtering by row
            weight = len(filtered_df)/len(merged)
            subentropies_sum+=weight*entropy(filtered_df[Y])
        
        # TODO: Return information gain
        return entropy(Y) - subentropies_sum
    



    def _gini_Split(self, feature, X, Y): #feature is just the name of feature.
        
        def gini(y):  #y is a series of label given as an input
           # TODO: Calculate gini impurity for target variable
            #1- sigma(p(i)**2)
            total_count = len(y)
            yes_counts = (y == "satisfied").sum()
            fraction = yes_counts/total_count
            result = 1-(fraction**2 + (1-fraction)**2)
            return result

        desired_column = X[feature]
        merged = pd.concat([desired_column,Y],axis=1)
        
        # Sum of the weighted gini's for sub_dataframes which is a rowly filtered version of main dataframe given as input
        #  based on the unique answers of the feature given again as the input :
        Sub_gini_sum = 0
        for value in desired_column.unique():
            # TODO: Calculate weighted gini for each value
            filtered_df = merged[merged[feature]==value]
            weight  =len(filtered_df)/len(merged)
            Sub_gini_sum += weight*gini(filtered_df[Y])
            

        # TODO: Return gini split value
        return gini(Y) - Sub_gini_sum

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