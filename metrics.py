from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    data_size = y.size #size of the dataset
    if data_size == 0:
        raise ValueError("Dataset is empty, accuracy cannot be calculated!")
    correct_predictions = (y_hat==y).sum()
    accuracy_val = correct_predictions / data_size
    return accuracy_val


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    assert cls in y.unique() # To check if the specified class is present in the unique values of y
    #getting true positive and false positive instances
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_positives = ((y_hat == cls) & (y != cls)).sum()
    total_positives = true_positives + false_positives
    #no predicted positives
    if total_positives == 0:
        return 1
    precision_val = true_positives/total_positives
    return precision_val


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    assert cls in y.unique()
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    false_negatives = ((y_hat != cls) & (y == cls)).sum()

    total_actual_positives = true_positives + false_negatives

    if total_actual_positives == 0:
        raise ValueError("no actual instances of class in the dataset")

    recall_val = true_positives / total_actual_positives

    return recall_val


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    data_size = y.size 

    if data_size == 0:
        raise ValueError("Data set is empty, RMSE cannot be calculated")

    mse_val = np.mean((y_hat - y) ** 2)
    rmse_val = np.sqrt(mse_val)

    return rmse_val


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    absolute_errors = (abs(y_hat - y)).sum()
    mae_val = absolute_errors.mean()
    return mae_val
