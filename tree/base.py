"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import one_hot_encoding, check_ifreal, entropy, gini_index, information_gain, opt_split_attribute, split_data

np.random.seed(42)

@dataclass
class Node:
    def __init__(self, split_column=None, val=None, depth=None):
        self.value = val  # Value assigned to the node (predicted output)
        self.split_column = split_column  # Feature index used for splitting at this node
        self.split_value = None  # Threshold value for the split (only in case of real o/p)
        self.children = {}  # Dictionary to store child nodes
        self.freq = None  # Frequency of occurrences of the node in the training data
        self.depth = depth  # Depth of the node in the tree

    # To get the predicted value for a given input instance
    def node_val(self, X, max_depth=np.inf):
        # Check if the node is a leaf or the maximum depth is reached
        if self.depth >= max_depth or self.split_column is None:
            return self.value
        else:
            if self.split_value is None:
                # Categorical split: check if the category exists in children
                curr_split = self.split_column
                if X[curr_split] in self.children:
                    # Recursively call node_val for the corresponding child
                    return self.children[X[curr_split]].node_val(X.drop(curr_split), max_depth=max_depth)
                else:
                    return self.value
            else:
                # Numerical split: check if the value is greater than the threshold
                curr_split = self.split_column
                if X[curr_split] > self.split_value:
                    return self.children["right"].node_val(X, max_depth=max_depth)
                else:
                    return self.children["left"].node_val(X, max_depth=max_depth)

    # To print the tree structure recursively
    def print_tree(self, space=1):
        # Check if the node is a leaf
        if self.split_column is None:
            print(" Value: {:.2f}, Depth: {}".format(self.value, self.depth), end="")
            return
        else:
            # Check if the split is numerical or categorical
            if self.split_value is not None:
                # Numerical split: print the condition and recursively call print_tree for children
                if space == 1:
                    print("?(X{} <= {:.2f})".format(self.split_column, self.split_value), end="")
                else:
                    print(" ?(X{} <= {:.2f})".format(self.split_column, self.split_value), end="")

                for i in self.children:
                    # Print Y for left child and N for right child
                    if i == "left":
                        print("\n" + ":  " * space + f"Y:", end="")
                    else:
                        print("\n" + ":  " * space + f"N:", end="")
                    self.children[i].print_tree(space=space + 1)
            else:
                # Categorical split: print conditions and recursively call print_tree for children
                for i in self.children:
                    print("\n" + ":  " * (space - 1) + f"?(X{self.split_column} = {i})", end="")
                    self.children[i].print_tree(space=space + 1)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="information_gain", max_depth=5):
        # Initialize the DecisionTree with specified criterion and maximum depth.
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.attribute = None
        self.output_type = None
        self.X_size = None

    def build_tree(self, X, Y, parent, depth=0):
        # Recursively build the decision tree based on the input data and labels.
        
        # Base case: If all labels are the same, create a leaf node with that label value.
        if Y.unique().size == 1:
            return Node(val=Y.values[0], depth=depth)

        # Recursive case: Continue splitting the data until a base case is reached or max depth is exceeded.
        if len(X.columns) > 0 and depth < self.max_depth and len(list(X.columns)) != sum(list(X.nunique())):
            split_col = None
            real_split_value = None

            # Find the optimal split attribute and its value based on the selected criterion.
            if self.criterion in ["information_gain", "gini_index"]:
                split_col = opt_split_attribute(X, Y, self.criterion, X.columns)
                _, real_split_value = information_gain(Y, X[split_col], self.criterion)

            # Create a node for the current split.
            curr_node = Node(split_column=split_col)
            root_col = X[split_col]

            # Handle categorical split column.
            if root_col.dtype.name == "category":
                X = X.drop(split_col, axis=1)
                root_classes = root_col.groupby(root_col).count()

                # Recursively build subtrees for each class in the categorical split.
                for class_type in list(root_classes.index):
                    curr_rows = (root_col == class_type)
                    if curr_rows.sum() > 0:
                        curr_node.children[class_type] = self.build_tree(X[curr_rows], Y[curr_rows], curr_node, depth=depth + 1)
                        curr_node.children[class_type].freq = len(X[curr_rows]) / self.X_size

            # Handle numerical split column.
            else:
                left_split, right_split = split_data(X, Y, split_col, real_split_value)
                curr_node.children["left"] = self.build_tree(left_split[0], left_split[1], curr_node, depth=depth + 1)
                curr_node.children["right"] = self.build_tree(right_split[0], right_split[1], curr_node, depth=depth + 1)
                curr_node.split_value = real_split_value

            # Set the value of the current node based on the label type.
            if Y.dtype.name == "category":
                curr_node.value = Y.mode(dropna=True)[0]
            else:
                curr_node.value = Y.mean()

            curr_node.depth = depth
            return curr_node

        else:
            # Create a leaf node with the majority class or mean value of the labels.
            if Y.dtype.name == "category":
                return Node(val=Y.mode(dropna=True)[0], depth=depth)
            else:
                return Node(val=Y.mean(), depth=depth)

    def fit(self, X, y): 
        # Fit the decision tree to the training data.
        self.output_type = y.dtype
        self.attribute = y.name
        self.X_size = len(X)
        self.root = self.build_tree(X, y, None)
        self.root.freq = 1

    def predict(self, X, max_depth=np.inf):
        # Predict the labels for the input data using the trained decision tree.
        Y = []
        for x in (X.index):
            Y.append(self.root.node_val(X.loc[x], max_depth=max_depth))
        Y_hat = pd.Series(Y, name=self.attribute).astype(self.output_type)
        return Y_hat

    def plot(self):
        # Plot the decision tree structure.
        self.root.print_tree(space=1)
        print('\n')
