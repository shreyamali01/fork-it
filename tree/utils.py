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

    return y.dtype.name == "float64"


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    value, counts = np.unique(Y, return_counts=True)
    total = counts.sum() 
    probabilities = counts / total
    entropy = 0
    for pi in probabilities:
        entropy -= pi * np.log2(pi)
    return entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    values, counts = np.unique(Y, return_counts = True)
    total = counts.sum()
    probabilities = counts / total

    gini_index = 1
    sum_of_squares = np.sum(np.square(probabilities))
    gini_index -= sum_of_squares
    return gini_index

def information_gain(Y, attr, criterion=None):
    # REAL INPUT REAL OUTPUT
    if (check_ifreal(Y) and check_ifreal(attr)):
        # Create a table with 'attribute' and 'output' columns
        table = pd.concat([attr, Y], axis=1).reindex(attr.index)
        table.columns = ['attribute', 'output']
        table.sort_values(by='attribute', inplace=True)
        input_values = table['attribute'].to_numpy()
        output_values = table['output'].to_numpy()

        optimum_split = 0
        max_gain = -np.inf
        parent_var = np.var(Y)

        # Iterate over potential split points to find the one with maximum gain
        for i in range(1, len(input_values)):
            curr_split = float(input_values[i] + input_values[i - 1]) / 2
            left_split = table[table['attribute'] <= curr_split]['output']
            right_split = table[table['attribute'] > curr_split]['output']

            child_var = 0
            child_var += (left_split.size / output_values.size) * (np.var(left_split))
            child_var += (right_split.size / output_values.size) * (np.var(right_split))
            split_var = parent_var - child_var

            if (split_var > max_gain):
                max_gain = split_var
                optimum_split = curr_split

        return max_gain, optimum_split

    # REAL INPUT DISCRETE OUTPUT USING INFORMATION GAIN
    elif (check_ifreal(attr) and (not check_ifreal(Y)) and criterion == "information_gain"):
        # Create a table with 'attribute' and 'output' columns
        table = pd.concat([attr, Y], axis=1).reindex(attr.index)
        table.columns = ['attribute', 'output']
        table.sort_values(by='attribute', inplace=True)
        input_values = table['attribute'].to_numpy()
        output_values = table['output'].to_numpy()

        optimum_split = 0
        max_gain = -np.inf
        parent_entropy = entropy(Y)

        # Iterate over potential split points to find the one with maximum gain
        for i in range(1, len(input_values)):
            curr_split = float(input_values[i] + input_values[i - 1]) / 2
            left_split = table[table['attribute'] <= curr_split]['output'].to_numpy()
            right_split = table[table['attribute'] > curr_split]['output'].to_numpy()
            child_entropy = 0
            child_entropy += (left_split.size / output_values.size) * (entropy(left_split))
            child_entropy += (right_split.size / output_values.size) * (entropy(right_split))
            split_entropy = parent_entropy - child_entropy

            if (split_entropy > max_gain):
                max_gain = split_entropy
                optimum_split = curr_split

        return max_gain, optimum_split

    # REAL INPUT DISCRETE OUTPUT USING GINI INDEX
    elif (check_ifreal(attr) and (not check_ifreal(Y)) and criterion == "gini_index"):
        # Create a table with 'attribute' and 'output' columns
        table = pd.concat([attr, Y], axis=1).reindex(attr.index)
        table.columns = ['attribute', 'output']
        table.sort_values(by='attribute', inplace=True)
        input_values = table['attribute'].to_numpy()
        output_values = table['output'].to_numpy()

        optimum_split = 0
        max_gain = -np.inf
        parent_gini = gini_index(Y)

        # Iterate over potential split points to find the one with maximum gain
        for i in range(1, len(input_values)):
            curr_split = float(input_values[i] + input_values[i - 1]) / 2
            left_split = table[table['attribute'] <= curr_split]['output'].to_numpy()
            right_split = table[table['attribute'] > curr_split]['output'].to_numpy()
            child_gini = 0
            child_gini += (left_split.size / output_values.size) * (gini_index(left_split))
            child_gini += (right_split.size / output_values.size) * (gini_index(right_split))
            split_gini = parent_gini - child_gini

            if (split_gini > max_gain):
                max_gain = split_gini
                optimum_split = curr_split

        return max_gain, optimum_split

    # DISCRETE INPUT DISCRETE OUTPUT USING INFORMATION GAIN
    elif ((not check_ifreal(attr)) and (not check_ifreal(Y)) and criterion == "information_gain"):
        parent_entropy = entropy(Y)
        total_size = Y.size
        classes = np.unique(attr)
        child_entropy = 0

        # Calculate child entropy for each class
        for i in classes:
            curr_class = Y[attr == i]
            class_entropy = entropy(curr_class)
            child_entropy += (curr_class.size / total_size) * class_entropy

        return parent_entropy - child_entropy, None

    # DISCRETE INPUT DISCRETE OUTPUT USING GINI INDEX
    elif ((not check_ifreal(attr)) and (not check_ifreal(Y)) and criterion == "gini_index"):
        parent_gini = gini_index(Y)
        total_size = Y.size
        classes = np.unique(attr)
        child_gini = 0

        # Calculate child gini index for each class
        for i in classes:
            curr_class = Y[attr == i]
            class_gini = gini_index(curr_class)
            child_gini += (curr_class.size / total_size) * class_gini

        return parent_gini - child_gini, None

    # DISCRETE INPUT REAL OUTPUT
    elif ((not check_ifreal(attr)) and check_ifreal(Y)):
        parent_var = np.var(Y)
        total_size = Y.size
        classes = np.unique(attr)
        child_var = 0

        # Calculate child variance for each class
        for i in classes:
            curr_class = Y[attr == i]
            class_variance = np.var(curr_class)
            child_var += curr_class.size / Y.size * class_variance

        return parent_var - child_var, None



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # Initialize variables to track maximum information gain and the corresponding attribute
    max_info_gain = -np.inf
    opt_attribute = None

    # Iterate through each attribute to find the one with the maximum information gain
    for attribute in features:
        # Calculate information gain for the current attribute
        curr_info_gain, _ = information_gain(y, X[attribute], criterion)
        
        # Update max_info_gain and opt_attribute if the current attribute has higher information gain
        if curr_info_gain > max_info_gain:
            max_info_gain = curr_info_gain
            opt_attribute = attribute

    return opt_attribute

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real-valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    # Initialize variables to track maximum information gain and the corresponding attribute
    max_info_gain = -np.inf
    opt_attribute = None

    # Iterate through each attribute to find the one with the maximum information gain
    for attribute in features:
        # Calculate information gain for the current attribute
        curr_info_gain, _ = information_gain(y, X[attribute], criterion)
        
        # Update max_info_gain and opt_attribute if the current attribute has higher information gain
        if curr_info_gain > max_info_gain:
            max_info_gain = curr_info_gain
            opt_attribute = attribute

    return opt_attribute

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Check if the attribute is real-valued or discrete
    if check_ifreal(X[attribute]):
        # For real-valued attributes, split the data based on the specified value
        left_split = X[X[attribute] <= value]
        right_split = X[X[attribute] > value]
    else:
        # For discrete attributes, split the data based on equality with the specified value
        left_split = X[X[attribute] == value]
        right_split = X[X[attribute] != value]

    # Get corresponding output values for the left and right splits
    left_output = y[left_split.index]
    right_output = y[right_split.index]

    return (left_split, left_output), (right_split, right_output)
