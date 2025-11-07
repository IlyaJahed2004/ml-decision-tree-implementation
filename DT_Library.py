import pandas as pd
import numpy as np
from enum import Enum
from math import log2


class Node():
    def __init__(self, feature=None, children=None,isleaf = False,parent=None,datasample_labels = None,answer=None,edge_value= None,threshold = None,gain=None,gini=None):   # changed: added gain and gini attributes
        # TODO: Initialize node attributes
        self.feature = feature
        self.datasample_labels = datasample_labels
        self.children = children                    #children should be given as node
        self.is_leaf =isleaf
        self.parent = parent
        self.answer = answer
        self.edge_value = edge_value     #for categorical splits
        self.threshold = threshold       #for numeric splits
        self.gain = gain                # changed: store gain
        self.gini = gini                # changed: store gini


class DecisionTree():
    def __init__(self, mode="gain", max_Depth=float("inf"), min_Samples=2,
                 pruning_threshold=None, TODO="Any desired hyperparameters"):
       # You can change all class properties for better performance!
       # TODO: Initialize tree hyperparameters and root node
        self.mode = mode
        self.max_Depth = max_Depth
        self.min_Samples = min_Samples
        self.pruning_threshold = pruning_threshold
        self.root = None

    #For assigning the Root to the real root node in this tree:
    def fit(self, X_Train, Y_Train):
        # Defensive: convert numpy to DataFrame if needed
        if isinstance(X_Train, np.ndarray):
            X_Train = pd.DataFrame(X_Train)
        if isinstance(Y_Train, pd.Series):
            Y_Train = Y_Train.map({"satisfied": 1, "dissatisfied": 0, 1:1, 0:0})
            Y_Train = Y_Train.to_numpy()        
        self.root = self._create_Tree(X_Train,Y_Train,depth= 0)
        return self.root
    

    # Y is the label column and X is the dataframe of the attributes
    def _create_Tree(self, X, Y,parent= None ,depth=0):

        # helper method for edge cases in recursion:(like the given method named :_calculate_Value)
        def plurality_value(Y):  #Y is actually a numpy array
            yes_counts = (Y == 1).sum()
            total_counts = len(Y)
            #1 is interpreted as satisfied and 0 as dissatisfied"
            return {1:yes_counts/total_counts , 0: (total_counts-yes_counts)/total_counts} #returning a dict of two probability.
            
        all_features =list(X.columns)
        num_Samples = len(Y)

        # --- edge cases before splitting ---
        if len(all_features) == 0 or num_Samples == 0 or np.all(Y==1) or np.all(Y==0):
            if len(all_features) == 0 or num_Samples == 0:
                result = plurality_value(Y if num_Samples>0 else parent.datasample_labels)
                probable_label = max(result , key=result.get)
                return Node(isleaf=True, datasample_labels=Y,parent=parent,answer=probable_label)
            if np.all(Y==1):
                return Node(isleaf=True,answer=1,parent=parent,datasample_labels=Y)
            if np.all(Y==0):
                return Node(isleaf=True,answer=0,parent=parent,datasample_labels=Y)

        # Get best feature node
        best_Feature_node = self._get_best_Feature(X,Y)
        best_Feature_node.parent = parent

        # --- compute x for pruning check ---
        if self.mode == "gain":
            x = best_Feature_node.gain
        else:
            x = best_Feature_node.gini

        # --- main stopping condition including pruning threshold --- # changed
        if num_Samples < self.min_Samples or depth >= self.max_Depth or (self.pruning_threshold is not None and ((self.mode=="gain" and x <= self.pruning_threshold) or (self.mode=="gini" and x >= self.pruning_threshold))):
            result = plurality_value(Y)
            probable_label = max(result, key=result.get)
            return Node(isleaf=True, answer=probable_label, datasample_labels=Y,parent=parent)

        best_Feature_node_children = []

        current_threshold = best_Feature_node.threshold

        if current_threshold is not None:
            featurename = best_Feature_node.feature
            X = X.reset_index(drop=True)
            local_Y = pd.Series(Y, name='target')
            merged = pd.concat([X, local_Y], axis=1)

            # Separate mask based on threshold
            below_mask = merged[featurename] <= current_threshold
            above_mask = merged[featurename] > current_threshold

            # Sub-DataFrames for features (without current feature) 
            below_X = merged.loc[below_mask].drop(columns=[featurename, 'target'])
            above_X = merged.loc[above_mask].drop(columns=[featurename, 'target'])

            # Corresponding labels
            below_Y = merged.loc[below_mask, 'target'].to_numpy()
            above_Y = merged.loc[above_mask, 'target'].to_numpy()

            # Create child nodes recursively
            leftchild = self._create_Tree(below_X, below_Y, parent=best_Feature_node, depth=depth+1)
            rightchild = self._create_Tree(above_X, above_Y, parent=best_Feature_node, depth=depth+1)

            # Assign edge values to children
            leftchild.edge_value = f"<= {current_threshold}"
            rightchild.edge_value = f"> {current_threshold}"

            best_Feature_node_children.append(leftchild)
            best_Feature_node_children.append(rightchild)
            best_Feature_node.children = best_Feature_node_children
            return best_Feature_node

        else:
            for unique_val in best_Feature_node.children:
                featurename= best_Feature_node.feature
                local_Y = pd.Series(Y,name='target')
                X = X.reset_index(drop=True)
                merged = pd.concat([X,local_Y],axis=1)
                filtered_by_child = merged[merged[featurename]==unique_val]

                if filtered_by_child.empty:
                    continue 

                new_x = filtered_by_child.drop(columns=[featurename, 'target'])
                new_y = filtered_by_child['target'].to_numpy()

                if len(np.unique(new_y)) == 1:
                    leaf = Node(
                        isleaf=True,
                        answer=new_y[0],
                        datasample_labels=new_y,
                        parent=best_Feature_node,
                        edge_value=unique_val
                    )
                    best_Feature_node_children.append(leaf)
                    continue 
                else:
                    childnode = self._create_Tree(X=new_x,Y=new_y,parent=best_Feature_node,depth=depth+1)
                    childnode.edge_value = unique_val
                    best_Feature_node_children.append(childnode)

            best_Feature_node.children = best_Feature_node_children
            return best_Feature_node


    # X is dataframe of features,Y is series of labels.
    def _get_best_Feature(self, X, Y): 
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(Y, pd.Series):
            local_Y = Y.copy()
        elif isinstance(Y, np.ndarray):
            local_Y = pd.Series(Y)
        else:
            local_Y = pd.Series(Y)

        local_Y = local_Y.reset_index(drop=True)

        best_feature = None
        best_threshold = None
        gain = None
        gini = None

        if self.mode == "gain":
            max_val = float('-inf')
            for feature in list(X.columns):
                info_gain, threshold = self._information_Gain(feature, X, local_Y)
                if info_gain > max_val:
                    max_val = info_gain
                    best_feature = feature
                    best_threshold = threshold
            gain = max_val

        elif self.mode == "gini":
            min_val = float('inf')
            for feature in list(X.columns):
                gini_score, threshold = self._gini_Split(feature, X, local_Y)
                if gini_score < min_val:
                    min_val = gini_score
                    best_feature = feature
                    best_threshold = threshold
            gini = min_val

        if best_feature is None:
            return Node(feature=None, datasample_labels=Y, children=[])

        if best_threshold is not None:
            return Node(feature=best_feature, datasample_labels=Y, children=[f"<={best_threshold}", f">{best_threshold}"], threshold=best_threshold,gain=gain,gini=gini)
        else:
            bestfeature_serie = X[best_feature].reset_index(drop=True)
            merged_with_label = pd.concat([bestfeature_serie, local_Y], axis=1)
            children_list = []
            for uniquedgevalueal in merged_with_label[best_feature].unique().tolist():
                children_list.append(uniquedgevalueal)
            return Node(feature=best_feature, datasample_labels=Y, children=children_list,gain=gain,gini=gini)


    def _information_Gain(self, feature, X, Y): 
        if ((isinstance(Y,pd.Series)) and (Y.name is None)):
            Y = Y.rename("target")
        if isinstance(Y,np.ndarray):
            Y_np_local = Y.copy()
        elif not isinstance(Y, np.ndarray):
            Y_np_local = Y.map({"satisfied": 1, "dissatisfied": 0})
            Y_np_local = Y.to_numpy() 
    
        def entropy(y):   
            if len(y) == 0:
                return 0.0
            yes_counts = (y == 1).sum()
            total_counts = len(y)
            fraction = yes_counts / total_counts
            if fraction == 0 or fraction == 1:
                return 0
            result = -fraction * log2(fraction) - (1 - fraction) * log2(1 - fraction)
            return result

        desired_column = X[feature]  
        parent_entropy = entropy(Y_np_local)

        if(pd.api.types.is_numeric_dtype(desired_column) and len(desired_column.dropna().unique()) >= 5):  
            best_gain=float("-inf")
            best_threshold= None
            sortedserie = np.sort(desired_column.dropna().unique()) 
            candidate_thresholds =(sortedserie[:-1] + sortedserie[1:])/2
            for threshold in candidate_thresholds:
                left_part = desired_column<=threshold
                right_part = desired_column>threshold
                left_y = Y_np_local[left_part.to_numpy()] if isinstance(left_part, pd.Series) else Y_np_local[left_part]
                right_y = Y_np_local[right_part.to_numpy()] if isinstance(right_part, pd.Series) else Y_np_local[right_part]  
                if(len(left_y)==0 or len(right_y)==0):
                    continue
                weight_left = len(left_y)/len(Y_np_local)
                weight_right = len(right_y)/len(Y_np_local)
                gain = parent_entropy - (weight_left * entropy(left_y) + weight_right * entropy(right_y))
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
            if best_threshold is None:    
                return 0, None
            return best_gain, best_threshold    
        else:
            desired_nparray = desired_column.to_numpy() 
            merged = np.column_stack((desired_nparray, Y))   
            subentropies_sum = 0
            for value in pd.unique(desired_nparray):  # <- changed here:it works both with Pandas Series and NumPy arrays
                filtered_np = merged[merged[:,0] == value] 
                weight = len(filtered_np) / len(merged)
                subentropies_sum += weight * entropy(filtered_np[:,1])
            return float(parent_entropy - subentropies_sum) , None


    def _gini_Split(self, feature, X, Y): 
        if ((isinstance(Y,pd.Series)) and (Y.name is None)):
            Y = Y.rename("target")
        if isinstance(Y,np.ndarray):
            Y_np_local = Y.copy()
        elif not isinstance(Y, np.ndarray):
            Y_np_local = Y.map({"satisfied": 1, "dissatisfied": 0})
            Y_np_local = Y_np_local.to_numpy()    

        def gini(y):  
            if len(y) == 0:
                return 0.0
            total_count = len(y)
            yes_counts = (y == 1).sum()
            fraction = yes_counts / total_count
            if fraction == 0 or fraction == 1:
                return 0
            result = 1 - (fraction ** 2 + (1 - fraction) ** 2)
            return result

        desired_column = X[feature]     

        if(pd.api.types.is_numeric_dtype(desired_column) and len(desired_column.dropna().unique()) >= 5):  
            best_score = float("inf")
            best_thr = None
            sortedserie = np.sort(desired_column.dropna().unique()) 
            candidate_thresholds = (sortedserie[:-1] + sortedserie[1:]) / 2
            for threshold in candidate_thresholds:
                left_part = desired_column <= threshold
                right_part = desired_column > threshold
                left_y = Y_np_local[left_part.to_numpy()] if isinstance(left_part, pd.Series) else Y_np_local[left_part]
                right_y = Y_np_local[right_part.to_numpy()] if isinstance(right_part, pd.Series) else Y_np_local[right_part]
                if(len(left_y) == 0 or len(right_y) == 0):
                    continue
                weight_left = len(left_y) / len(Y_np_local)
                weight_right = len(right_y) / len(Y_np_local)
                score = weight_left * gini(left_y) + weight_right * gini(right_y)
                if score < best_score:
                    best_score = score
                    best_thr = threshold
            if best_thr is None:
                return float("inf"), None
            return best_score, best_thr
        else:
            desired_nparray = desired_column.to_numpy()  
            merged = np.column_stack((desired_nparray, Y_np_local))  
            Sub_gini_sum = 0
            for value in pd.unique(desired_nparray):  # <- changed here:it works both with Pandas Series and NumPy arrays
                filtered_df = merged[merged[:,0] == value]
                weight = len(filtered_df) / len(merged)
                Sub_gini_sum += weight * gini(filtered_df[:,1]) 
            return float(Sub_gini_sum), None



    def predict(self, X):
        """
        Predict labels for rows in X.
        Accepts: pandas.DataFrame, pandas.Series (single row), or numpy 2D array.
        Returns: numpy.ndarray of int labels.
        """

        # this is a helper method for plurality.here we have a node and based on the labels it has, we return the value with maximum counts.
        def plurality_label_from_node(node):
            array = node.datasample_labels
            if array is None:
                return 0
            
            array = np.asarray(array)
            if array.size == 0:
                return 0
            
            yes_counts = 0
            for x in array:
                if x == 1:
                    yes_counts += 1
            
            no_counts = len(array) - yes_counts
            return 1 if yes_counts > no_counts else 0



        def _move_Tree(sample, node):

            # defensive
            if node is None:
                return 0

            if (node.is_leaf):
                return int(node.answer)

            feature = node.feature
            if feature is None:
                return plurality_label_from_node(node)

            # sample may be missing the feature
            if feature not in sample:
                return plurality_label_from_node(node)

            val = sample[feature]

            threshold = node.threshold
            # numeric split (threshold present)
            if threshold is not None:
                # expected children: [left_child, right_child]
                children = node.children
                if not children or len(children) < 2:
                    return plurality_label_from_node(node)
                try:
                    # handle NaN or non-comparable => fallback
                    go_left = (val <= threshold)
                except Exception:
                    return plurality_label_from_node(node)

                child = children[0] if go_left else children[1]
                if child is None:
                    return plurality_label_from_node(node)
                return _move_Tree(sample, child)

            # categorical split: match child.edge_value to sample value
            children = node.children
            if not children:
                return plurality_label_from_node(node)

            for child in children:
                # child's edge_value should hold the category that leads to that child
                edgevalue = child.edgevalue
                #  compare with exact match or string-equal 
                if edgevalue == val or (edgevalue is not None and str(edgevalue) == str(val)):
                    return _move_Tree(sample, child)

            # unseen category -> plurality fallback
            return plurality_label_from_node(node)

        # --- input normalization ---
        if self.root is None:
            # no tree fitted -> return zeros
            if X is None:
                return np.array([], dtype=int)
            if isinstance(X, (pd.DataFrame, pd.Series)):
                n = len(X) if isinstance(X, pd.DataFrame) else 1
                return np.zeros(n, dtype=int)
            arr = np.asarray(X)
            if arr.ndim == 1:
                return np.zeros(1, dtype=int)
            return np.zeros(arr.shape[0], dtype=int)

        # convert inputs to a DataFrame of rows
        if isinstance(X, np.ndarray):
            X_proc = pd.DataFrame(X)
        elif isinstance(X, pd.Series):
            X_proc = X.to_frame().T.reset_index(drop=True)
        else:
            X_proc = X.reset_index(drop=True)

        preds = []
        for _, row in X_proc.iterrows():
            sample_dict = row.to_dict()
            try:
                p = _move_Tree(sample_dict, self.root)
            except Exception:
                # worst-case fallback to root plurality
                p = plurality_label_from_node(self.root)
            preds.append(int(p))

        return np.array(preds, dtype=int)

