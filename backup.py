import pandas as pd
import numpy as np
from enum import Enum
from math import log2


class Node():
    def __init__(self, feature=None, children=None,isleaf = False,parent=None,datasample_labels = None,answer=None,edge_value= None,threshold = None):   # the edge value is the edge from a node's parent to itself.
        # TODO: Initialize node attributes
        self.feature = feature
        self.datasample_labels = datasample_labels
        self.children = children                    #children should be given as node
        self.is_leaf =isleaf
        self.parent = parent
        self.answer = answer
        self.edge_value = edge_value     #for categorical splits
        self.threshold = threshold       #for numeric splits


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


        # Check stopping conditions (Pre-Pruning)
        if num_Samples >= self.min_Samples and depth < self.max_Depth:

            best_Feature_node_children = []

            if(len(all_features)==0):
                result = plurality_value(Y)
                probable_label = max(result, key=result.get)
                leaf_node = Node(isleaf= True,datasample_labels= Y,parent= parent,answer= probable_label)
                return leaf_node
            
            elif(X.shape[0] == 0): #when there are no data in the df.
                # parent_node: i manipulated the node class in a way that it containes the samples too in itself so i can do plurality on that.
                result = plurality_value(parent.datasample_labels)
                probable_label = max(result , key=result.get)
                leaf_node = Node(isleaf=True, datasample_labels=Y ,parent=parent,answer=probable_label)
                return leaf_node
            
            elif np.all(Y == 1):
                leafnode = Node(isleaf=True,answer=1, parent=parent,datasample_labels=Y)
                return leafnode
                
            elif np.all(Y==0):
                leafnode = Node(isleaf=True,answer=0, parent=parent,datasample_labels=Y)
                return leafnode
            
            else:

                best_Feature_node = self._get_best_Feature(X, Y)   #best-feature here is a node containing the label and the dataframe of the remained attributes.
                best_Feature_node.parent = parent
                
                current_threshold = best_Feature_node.threshold

                if(current_threshold is not None):
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
                        # TODO: Recursively create child nodes
                        # the child here is only one unique value of the values we have for the selected attribute
                        featurename= best_Feature_node.feature
                        # naming is used when deleting the labels from the new dataframe:
                        local_Y = pd.Series(Y,name='target')
                        X = X.reset_index(drop=True)
                        merged = pd.concat([X,local_Y],axis=1)
                        filtered_by_child = merged[merged[featurename]==unique_val]  #this is a selection by row.

                        if filtered_by_child.empty:  # so important to prevent irrational behavior because when there is no data it shouldnt step in the next steps in this iteration.
                            continue 

                        #new_x = remaining features only
                        new_x = filtered_by_child.drop(columns=[featurename, 'target'])

                        # new_y = label column
                        new_y = filtered_by_child['target']
                        new_y = new_y.to_numpy()

                        if len(np.unique(new_y)) == 1:    #we check this here to avoid making multiple nodes.
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
                            # Recursively calling create_tree for the new df and corresponding labels:
                            childnode = self._create_Tree(X= new_x, Y= new_y,parent= best_Feature_node,depth= depth+1)
                            #when we reach here it means that we havent faced edge cases:
                            childnode.edge_value = unique_val
                            best_Feature_node_children.append(childnode)
                            
                    best_Feature_node.children = best_Feature_node_children
                    return best_Feature_node
        
    # TODO: Create leaf node with predicted value
        else:
            result = plurality_value(Y)
            probable_label = max(result, key=result.get)
            return Node(isleaf=True, answer=probable_label, datasample_labels=Y,parent=parent)


    # X is dataframe of features,Y is series of labels.
    def _get_best_Feature(self, X, Y): 
        # Ensure X is a dataframe
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        # Ensure Y is a pandas Series
        if isinstance(Y, pd.Series):
            local_Y = Y.copy()
        elif isinstance(Y, np.ndarray):
            local_Y = pd.Series(Y)
        else:
            local_Y = pd.Series(Y)  # fallback

        # normalize index for safe boolean indexing later
        local_Y = local_Y.reset_index(drop=True)

        best_feature = None
        best_threshold = None

        if self.mode == "gain":
            # TODO: Calculate information gain for each feature
            max_val = float('-inf')
            for feature in list(X.columns):
                info_gain, threshold = self._information_Gain(feature, X, local_Y)
                if info_gain > max_val:
                    max_val = info_gain
                    best_feature = feature
                    best_threshold = threshold

        elif self.mode == "gini":
            # TODO: Calculate gini split for each feature
            min_val = float('+inf')
            for feature in list(X.columns):
                gini_score, threshold = self._gini_Split(feature, X, local_Y)
                if gini_score < min_val:
                    min_val = gini_score
                    best_feature = feature
                    best_threshold = threshold

        # If no feature was selected (e.g., empty X), return a fallback leaf-node-like Node
        if best_feature is None:
            return Node(feature=None, datasample_labels=Y, children=[])

        # If numeric split (threshold provided) -> binary children
        if best_threshold is not None:
            return Node(feature=best_feature, datasample_labels=Y, children=[f"<={best_threshold}", f">{best_threshold}"], threshold=best_threshold)

        else:
            # TODO: Select and return best feature as Node
            # for now i put list of names of the children features not a node:
            bestfeature_serie = X[best_feature].reset_index(drop=True)
            merged_with_label = pd.concat([bestfeature_serie, local_Y], axis=1)
            children_list = []
            for uniqueval in merged_with_label[best_feature].unique().tolist():
                children_list.append(uniqueval)
            return Node(feature=best_feature, datasample_labels=Y, children=children_list)


    def _information_Gain(self, feature, X, Y): 
        # X is the dataframe with all the feature columns.
        # Y is a series of the label (or numpy array when test are running.)

        if ((isinstance(Y,pd.Series)) and (Y.name is None)):
            Y = Y.rename("target")

        if isinstance(Y,np.ndarray):
            Y_np_local = Y.copy()
        elif not isinstance(Y, np.ndarray):
            Y_np_local = Y.map({"satisfied": 1, "dissatisfied": 0})
            Y_np_local = Y.to_numpy() 

        def entropy(y):   
            # TODO: Calculate entropy for target variable
            if len(y) == 0:
                return 0.0
            yes_counts = (y == 1).sum()
            total_counts = len(y)
            fraction = yes_counts / total_counts
            if fraction == 0 or fraction == 1:
                return 0
            result = -fraction * log2(fraction) - (1 - fraction) * log2(1 - fraction)
            return result

        desired_column = X[feature]  #this is a serie.
        parent_entropy = entropy(Y_np_local)

        # this is for numerical feature or numeric with <5 unique values treated as categorical
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
            # categorical or numeric with <5 unique values
            desired_nparray = desired_column.to_numpy() 
            merged = np.column_stack((desired_nparray, Y))   
            subentropies_sum = 0
            for value in np.unique(desired_nparray):
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
            Y_np_local = Y.to_numpy()    

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

        # numeric or numeric with <5 unique values treated as categorical
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
            for value in np.unique(desired_nparray):
                filtered_df = merged[merged[:,0] == value]
                weight = len(filtered_df) / len(merged)
                Sub_gini_sum += weight * gini(filtered_df[:,1]) 
            return float(Sub_gini_sum), None


    # def _calculate_Value(self, Y):
    #     # Where is it used and what does it do?
    #     return max(set(Y), key=list(Y).count) 



    def predict(self, X):
        """
        Predict labels for rows in X.
        Accepts: pandas.DataFrame, pandas.Series (single row), or numpy 2D array.
        Returns: numpy.ndarray of int labels.
        """

        # --- helpers ---
        def plurality_label_from_node(node):
            """Return majority class (int) using node.datasample_labels (numpy or list)."""
            arr = getattr(node, "datasample_labels", None)
            if arr is None:
                return 0
            arr = np.asarray(arr)
            if arr.size == 0:
                return 0
            uniq, counts = np.unique(arr, return_counts=True)
            return int(uniq[np.argmax(counts)])

        def _move_Tree(sample, node):
            """
            Recursive navigation:
            - sample: dict-like of feature->value
            - node: Node object
            Returns predicted label (int)
            """
            # defensive
            if node is None:
                return 0

            if getattr(node, "is_leaf", False):
                return int(node.answer)

            feat = getattr(node, "feature", None)
            if feat is None:
                return plurality_label_from_node(node)

            # sample may be missing the feature
            if feat not in sample:
                return plurality_label_from_node(node)

            val = sample[feat]

            # numeric split (threshold present)
            thr = getattr(node, "threshold", None)
            if thr is not None:
                # expected children: [left_child, right_child]
                children = getattr(node, "children", None)
                if not children or len(children) < 2:
                    return plurality_label_from_node(node)
                try:
                    # handle NaN or non-comparable => fallback
                    go_left = (val <= thr)
                except Exception:
                    return plurality_label_from_node(node)

                child = children[0] if go_left else children[1]
                if child is None:
                    return plurality_label_from_node(node)
                return _move_Tree(sample, child)

            # categorical split: match child.edge_value to sample value
            children = getattr(node, "children", None)
            if not children:
                return plurality_label_from_node(node)

            for child in children:
                # child's edge_value should hold the category that leads to that child
                ev = getattr(child, "edge_value", None)
                # tolerant compare: exact match or string-equal
                if ev == val or (ev is not None and str(ev) == str(val)):
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