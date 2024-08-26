"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """

    return pd.get_dummies(X,drop_first=True)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    return np.issubdtype(y.dtype,np.number)


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    #counting occurences of the classes
    value, counts = np.unique(Y, return_counts=True)

    #calculating probalities for each class
    prob_arr = counts/ len(Y)

    #calculating entropy
    entropy_val = -np.sum(prob_arr*np.log2(prob_arr + 1e-9))

    return entropy_val


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """

    #counting occurences of the classes 
    value, counts = np.unique(Y, return_counts=True)

    #calculating probalities for each class
    prob_arr = counts/ len(Y)

    #calculating gini index
    gini_val = 1 - np.sum(prob_arr**2)

    return gini_val

def mse(Y:pd.Series) -> float:
    """"
    Function to calculate mean square error (MSE)

    """

    mean_Y = np.mean(Y)

    mse_val = np.mean((Y-mean_Y)**2)

    return mse_val


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """

    #calculating initial impurity of the target

    if criterion == 'entropy':
        initial_impurity = entropy(Y)
    elif criterion == 'gini':
        initial_impurity = gini_index(Y)
    elif criterion == 'mse':
        initial_impurity = mse(Y)

    else:
        raise ValueError('Criterion must be either entropy, gini or mse')
    
    unique_values = attr.unique()
    weighted_impurity = 0

    for value in unique_values:
        subset_Y = Y[attr == value]
    
    #calculating entropy based on criterion
    if criterion == 'entropy':
        subset_impurity = entropy(subset_Y)
    elif criterion == 'gini':
        subset_impurity = gini_index(subset_Y)
    elif criterion == 'mse':
        subset_impurity = mse(subset_Y)

    weight = len(subset_Y) / len(Y)
    weighted_impurity += weight * subset_impurity

    info_gain = initial_impurity - weighted_impurity

    return info_gain


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    best_attr = None
    best_i_gain = -float('inf')
    initial_impurity = None

    #finding initial impurity based on criterion
    if criterion == 'entropy':
        initial_impurity = entropy(y)
    elif criterion == 'gini':
        initial_impurity = gini_index(y)
    elif criterion == 'mse':
        initial_impurity = mse(y)
    
    for attr in features:
        #calculating information gain for each attribute
        info_gain = information_gain(y, X[attr], criterion)
        
        #updating the best attribute if this one is better
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_attr = attr

    return best_attr

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # If the attribute is real-valued, split based on a threshold
    if np.issubdtype(X[attribute].dtype, np.number):
        left_mask = X[attribute] <= value
        right_mask = X[attribute] > value
    else:
        # For categorical features, split based on the exact match
        left_mask = X[attribute] == value
        right_mask = X[attribute] != value
    
    # Split the data into two subsets
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]
    
    return (X_left, y_left), (X_right, y_right)

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
