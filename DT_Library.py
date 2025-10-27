import pandas as pd
import numpy as np
from enum import Enum
from math import log2


class Node():
    def __init__(self, feature=None, children=None,isleaf = False,parent=None,datasample_labels = None,answer=None TODO="Other information about that node you wana show and need."):
        # TODO: Initialize node attributes
        self.feature = feature
        self.datasamples_labels = datasample_labels
        self.children = children
        self.is_leaf =isleaf
        self.parent = parent
        self.answer = answer


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
        all_features =list(X.columns)
        num_Samples = len(Y)

        #helper method for edge cases in recursion:
        def plurality_value(Y):   # Y is actually a numpy array
            yes_counts = (Y == 1).sum()
            total_counts = len(Y)
            return {"Yes":yes_counts/total_counts , "No": total_counts-yes_counts/total_counts} #returning a dict of two probability.
        

        # Check stopping conditions (Pre-Pruning)
        if num_Samples >= self.min_Samples and depth < self.max_Depth:
            best_Feature_node = self._get_best_Feature(X, Y)   #best-feature here is a node containing the label and the dataframe of the remained attributes.

            all_features.remove(best_Feature_node.feature)

            # children = []
            # children = best_Feature.children

            if(len(all_features)==0):
                best_Feature_node.is_leaf=True
                parentnode = best_Feature_node.parent
                result = plurality_value(parentnode.datasample_labels)
                probable_label = max(result, key=result.get)
                best_Feature_node.answer = probable_label
                return best_Feature_node

            elif((Y=="satisfied").all()):
                best_Feature_node.is_leaf =True
                best_Feature_node.answer = "satisfied"
                return best_Feature_node
            
            elif((Y=="dissatisfied").all()):
                best_Feature_node.is_leaf =True
                best_Feature_node.answer = "dissatisfied"
                return best_Feature_node

            elif(X.empty()):
                # parent_node: i manipulated the node class in a way that it containes the samples too in itself so i can do plurality on that.
                best_Feature_node.is_leaf=True
                parentnode = best_Feature_node.parent
                result = plurality_value(parentnode.datasample_labels)
                probable_label = max(result , key=result.get)
                return best_Feature_node


            # Check gain or gini!
            for child in best_Feature_node.children:
                # TODO: Recursively create child nodes
                # the child here is only one unique value of the values we have for the selected attribute.i should retireve a dataframe and serie of label for this:
                merged = pd.concatenate([X,Y],axis=1)
                merged = merged[merged[best_Feature_node.feature]==child]  #this is a selection by row.

                # this raw dataframe is for features and their datas not the labels and not the bestfeature column either:
                df_raw = merged.loc[:, merged.columns != Y.name and merged.columns != best_Feature_node.feature]

                # Recursively calling create_tree for the new df and corresponding labels:
                self._create_Tree(df_raw, merged[Y.name])

        # TODO: Create leaf node with predicted value
        return Node()


    # X is dataframe of features,   Y is series of labels.
    def _get_best_Feature(self, X, Y):                                  #i should manipulate this to return the datasample_labels of the bestfeature too.
        datasample_labels = None
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
        return Node(feature=best_feature,datasample_labels=Y,children=children_list)
    

    # infogain is defined for a feature.
    # feature is given to the method as string.(It is the name of the feature)
    def _information_Gain(self, feature, X, Y): #X is the dataframe with all the feature columns.    #Y is a series of the label (ir numpy array when test are running.)

        if (isinstance(Y,pd.Series) and (Y.name is None)):
            Y = Y.rename("target") 

        # Make sure Y is converted to numpy array just for optimization cases:

        if not isinstance(Y, np.ndarray): # if it is not instance of numpy array we know it is a pandas serie
            Y = Y.map({"satisfied": 1, "dissatisfied": 0, 1:1, 0:0})
            Y = Y.to_numpy()

       
        def entropy(y):   #in this version of code y is a numpy array of labels passed with the values "0" and "1"
           # TODO: Calculate entropy for target variable
            yes_counts = (y == 1).sum()
            total_counts = len(y)
            fraction = yes_counts / total_counts
            if fraction == 0 or fraction == 1:
                return 0
            result = -fraction * log2(fraction) - (1 - fraction) * log2(1 - fraction)
            return result

        desired_column = X[feature]  #this is a serie.
        desired_nparray = desired_column.to_numpy() # convert the serie to a numpy array
        merged = np.column_stack((desired_nparray, Y))   # it concatenates two numpy array like this:it assumes two arrays like columns and concatenates by row.

        
        # calculating the sum of the entropies for each sub_dataframes filtered by different answers for that attribute:
        subentropies_sum = 0
        for value in np.unique(desired_nparray):  #returns an array of the unique values that appear in that numpy array
            # TODO: Calculate weighted entropy for each value
            filtered_np = merged[merged[:,0] == value] #filtering by row for np arrays
            weight = len(filtered_np) / len(merged)
            subentropies_sum += weight * entropy(filtered_np[:,1])
        

        # TODO: Return information gain
        return entropy(Y) - subentropies_sum
    

    def _gini_Split(self, feature, X, Y): 

        # keyvalue error handling:
        if ((isinstance(Y,pd.Series)) and (Y.name is None)):
            Y = Y.rename("target")

        if not isinstance(Y, np.ndarray):
            Y = Y.map({"satisfied": 1, "dissatisfied": 0})
            Y = Y.to_numpy()     


        def gini(y):  #y is an nparray passed to gini method
           # TODO: Calculate gini impurity for target variable
            #1- sigma(p(i)**2)
            total_count = len(y)
            yes_counts = (y == 1).sum()
            fraction = yes_counts / total_count
            if fraction == 0 or fraction == 1:
                return 0
            result = 1 - (fraction ** 2 + (1 - fraction) ** 2)
            return result

        desired_column = X[feature]   #we get a serie here.
        desired_nparray = desired_column.to_numpy()  #converted the serie to a nparray.
        merged = np.column_stack((desired_nparray,Y))  #we have a 2-dimenstional array.the subarrays first element is for features values and the second element is for corresponding label.
        

        # Sum of the weighted gini's for sub_dataframes which is a rowly filtered version of main dataframe given as input
        #  based on the unique answers of the feature given again as the input :
        Sub_gini_sum = 0
        for value in np.unique(desired_nparray):
            # TODO: Calculate weighted gini for each value
            filtered_df = merged[merged[:,0] == value]
            weight = len(filtered_df) / len(merged)
            Sub_gini_sum += weight * gini(filtered_df[:,1]) 
            
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



#  test case i used for debugging:

# Y_test = pd.Series([0, 0, 1, 1, 0, 0, 0], name='target')
# X_test = pd.DataFrame({
#     'perfect_split': [0, 0, 1, 0, 2, 2, 2],
#     'other': [10, 20, 30, 40, 10, 5, 15]
# })
# feature_test = 'perfect_split'
# dt = DecisionTree(mode="gain")
# result = dt._information_Gain(feature_test, X_test, Y_test)
# print(result)
