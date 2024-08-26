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
from tree.utils import one_hot_encoding, check_ifreal, entropy, gini_index, mse, information_gain, opt_split_attribute, split_data

np.random.seed(42)

@dataclass
class Node:
    def __init__(self, attribute = None, children=None, value = None, is_leaf=False):
        self.attribute = attribute
        self.children = children if children is not None else {}
        self.value = value
        self.is_leaf = is_leaf


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        X_encoded = one_hot_encoding(X)

        if check_ifreal(y):
            self.criterion = 'mse'
        else:
            self.criterion = 'entropy' #default to entropy for categorical data
        
        #calling the recurssive fitting function
        self.root = self._fit_recursive(X_encoded, y, self.criterion)
    
    def _fit_recursive(self, X: pd.DataFrame, y: pd.Series, criterion: str) -> Node:
        #base case
        if len(np.unique(y)) == 1:
            return Node(value=y.mode()[0])
        
        #finding the best attribute to split
        best_attr = self.opt_split_attribute(X, y, self.criterion)

        #creating a root node for this subtree
        root_node = Node(attribute=best_attr)

        #obtaining unique values of the best attribute
        unique_values = X[best_attr].unique()

        for value in unique_values:
            #spliting the data based on the best attribute
            subset_X = X[X[best_attr] == value].drop(columns=best_attr)
            subset_y = y[X[best_attr] == value]
        
            #recursively fitting the subtree
            child_node = self._fit_recursive(subset_X, subset_y)
        
            #adding the child node to the root node's children
            root_node.children[value] = child_node

        return root_node

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        #initializing a list to store the predicted output values
        predictions = []

        #iterating over each row in the input DataFrame
        for _, row in X.iterrows():
            #starting at the root node
            current_node = self.root

            #traversing the tree until leaf node is reached
            while not current_node.is_leaf:
                if current_node.split_val is None:
                    current_node = current_node.child_.get(row[current_node.col_used], current_node)
                else:
                    if row[current_node.col_used] <= current_node.split_val:
                        current_node = current_node.child_["left"]
                    else:
                        current_node = current_node.child_["right"]
            predictions.append(current_node.val)

        return pd.Series(predictions, name=self.Y_name).astype(self.y_type)

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        def _plot(node, indent=""):
            if node.is_leaf:
                print(f"{indent}Y: Class {node.val}")
            else:
                if node.split_val is not None:
                    print(f"{indent}?(X{node.col_used} > {node.split_val})")
                    print(f"{indent}Y:")
                    _plot(node.child_["left"], indent + "    ")
                    print(f"{indent}N:")
                    _plot(node.child_["right"], indent + "    ")
                else:
                #in case of a categorical split
                    print(f"{indent}?(X{node.col_used})")
                    for value, child_node in node.child_.items():
                        print(f"{indent}Value {value}:")
                        _plot(child_node, indent + "    ")

    #plotting
        _plot(self.root)
