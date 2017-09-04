# -*- coding: utf-8 -*-

__author__ = "Takayuki Osogami"
__copyright__ = "(C) Copyright IBM Corp. 2016"

from .. import arraymath as amath


def MSE(y_true, y_pred):
    """
    Mean squared error of a sequence of predicted vectors

    y_true : array, shape(L, N)
    y_pred : array, shape(L, N)

    mean of (dy_1^2 + ... + dy_N^2 ) over L pairs of vectors
    (y_true[i], y_pred[i])
    """
    MSE_each_coordinate = amath.mean_squared_error(y_true, y_pred,
                                                   multioutput="raw_values")
    return amath.sum(MSE_each_coordinate)


def RMSE(y_true, y_pred):
    """
    Root mean squared error of a sequence of predicted vectors

    y_true: array, shape(L, N)
    y_pred: array, shape(L, N)

    squared root of the mean of (dy_1^2 + ... + dy_N^2 ) over L pairs of
    vectors (y_true[i], y_pred[i])
    """
    return amath.sqrt(MSE(y_true, y_pred))


def baseline_RMSE(init_pred, sequence):
    """
    Baseline RMSE where predictions are made using the previous observation.

    Parameters
    ----------
    init_pred : float or array, length n_dim
        prediction made at time step 0.
    sequence : list or generator
        time series used for performance evaluation.

    Returns
    -------
    float
        RMSE of the baseline method, forecasting using the previous observation.
    """
    last_pattern = [init_pred] + list(sequence[:-1])
    last_pattern = amath.array(last_pattern)
    baseline = RMSE(sequence, last_pattern)
    return baseline
