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
    return (y_hat==y).sum()/len(y)


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    true_positive=((y_hat==cls)&(y==clas)).sum()
    predicted_postive=(y_hat==cls).sum()
    return true_positive/predicted_positive if predicted_positive!=0 else 0.0
    

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    true_positive=((y_hat==cls)&(y==cls)).sum()
    actual_positive=(y==cls).sum()
    return true_positive/actual_positive if actual_postive !=0 else 0.0


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    return np.sqrt(((y_hat-y)**2).mean())


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    return np.abs(y_hat-y).mean()
