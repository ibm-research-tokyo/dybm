# (C) Copyright IBM Corp. 2016
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.


__author__ = "Takayuki Osogami"


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
