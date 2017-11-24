# (C) Copyright IBM Corp. 2017
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

""" Implementation of an Echo State Network
"""

__author__ = "Takayuki Osogami"

import numpy as np

from .. import arraymath as amath
from ..time_series.time_series_model import TimeSeriesModel
from ..base.sgd import AdaGrad


class ESN(TimeSeriesModel):

    """
    An Echo State Network

    Parameters
    ----------
    in_dim : int
        dimension of input time-series
    out_dim : int, optional
        dimension of target time-series
    esn_dim : int
        number of ESN internal units
    SGD : object of SGD.SGD, optional
        object of a stochastic gradient method
    L1 : float, optional
        strength of L1 regularization
    L2 : float, optional
        strength of L2 regularization
    spectral_radius : float, optional
        spectral radius of ESN internal weight matrix
    sparsity : float, optional
        sparsity of ESN internal connection
    leak : float, optional
        leak parameter of ESN

    Attributes
    ----------
    in_dim : int
        in_dim
    out_dim : int
        out_dim
    esn_dim : int
        esn_dim
    spectral_radius : float
        spectral_radius
    sparsity : float
        sparsity
    leak : float
        leak
    """

    def __init__(self, esn_dim, in_dim, out_dim=None, SGD=None, L1=0.,
                 L2=0., spectral_radius=0.95, sparsity=0., leak=1.0):
        if out_dim is None:
            out_dim = in_dim

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.esn_dim = esn_dim

        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak = leak

        self.random = amath.random.RandomState(0)

        self.variables = dict()

        # default setting of output matrix to zero
        self.variables["A"] = amath.zeros((self.esn_dim, self.out_dim),
                                          dtype=float)

        # initialize ESN internal weight matrix at edge of chaos
        self.Wrec = self.random.normal(0, 1, (self.esn_dim, self.esn_dim))
        # delete the fraction of connections given by (self.sparsity):
        amath.assign_if_true(
            self.Wrec, self.random.rand(*self.Wrec.shape) < self.sparsity, 0)
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eig(amath.to_numpy(self.Wrec))[0]))
        # rescale them to reach the requested spectral radius:
        self.Wrec = self.Wrec * self.spectral_radius / radius

        # initialise input weight matrix to ESN
        self.Win \
            = self.random.normal(0, 0.1, (self.in_dim, self.esn_dim)) / np.sqrt(self.in_dim)

        self.init_state()

        if SGD is None:
            SGD = AdaGrad()
        self.SGD = SGD.set_shape(self.variables)
        self.L2 = dict()
        self.L1 = dict()
        for key in self.variables:
            self.L1[key] = L1
            self.L2[key] = L2

        TimeSeriesModel.__init__(self)

    def init_state(self):

        # Internal state vector
        self.si = amath.zeros((self.esn_dim,))

    def _update_state(self, in_pattern):
        """Updating internal state

        Parameters
        ----------
        in_pattern : array, or list of arrays
            pattern used to update the state
        """
        new_si = amath.tanh(in_pattern.dot(self.Win) + self.si.dot(self.Wrec))
        self.si = (1 - self.leak) * self.si + self.leak * new_si

    def _get_delta(self, out_pattern, expected=None):
        """Getting how much we change parameters by learning a given pattern

        Parameters
        ----------
        out_patten : array, or list of arrays
            given target pattern
        expected : array, or list of arrays, optional
            expected pattern
        """
        assert out_pattern.shape == (self.out_dim,), \
            "out_pattern must have shape (out_dim,)"
        if expected is not None:
            assert expected.shape == (self.out_dim,), \
                "expected must have shape (out_dim,)"

        gradient = self._get_gradient(out_pattern, expected)
        self.SGD.update_state(gradient)
        delta = self.SGD.get_delta()

        return delta

    def _get_gradient(self, out_pattern, expected=None):
        """
        computing the gradient of log likelihood

        Parameters
        ----------
        out_pattern : array, length N
            out_pattern, where gradient is computed

        Returns
        -------
        gradient : dict of arrays
            dictionary with name of a variable as a key
        """
        assert out_pattern.shape == (self.out_dim,), \
            "out_pattern must have shape (out_dim,)"
        if expected is not None:
            assert expected.shape == (self.out_dim,), \
                "expected must have shape (out_dim,)"

        if expected is None:
            expected = self._get_mean()

        dx = out_pattern - expected

        gradient = dict()
        gradient["A"] = self.si.reshape((self.esn_dim, 1)) * dx

        assert gradient["A"].shape == (self.esn_dim, self.out_dim), \
            "gradient A must have shape (esn_dim, out_dim)"

        self.SGD.apply_L2_regularization(gradient, self.variables, self.L2)

        return gradient

    def _get_mean(self):
        """
        Computing estimated mean

        Returns
        -------
        mu : array, shape (out_dim,)
            estimated mean
        """
        mu = self.si.dot(self.variables["A"])

        assert mu.shape == (self.out_dim,), "mu must have shape (out_dim,)"

        return mu

    def _update_parameters(self, delta):
        """
        update the parameters by delta

        Parameters
        ----------
        delta : dict, or list of dicts
            amount by which the parameters are updated
        """
        self.SGD.update_with_L1_regularization(self.variables, delta, self.L1)

    def predict_next(self):
        """
        Predicting next pattern in a deterministic manner

        Returns
        -------
        array
            prediction
        """
        return self._get_mean()

    def _get_sample(self):
        return self._get_mean()

    def set_learning_rate(self, rate):
        """
        Setting the learning rate

        Parameters
        ----------
        rate : float
            learning rate
        """

        self.SGD.set_learning_rate(rate)

    def get_input_dimension(self):
        return self.in_dim

    def get_target_dimension(self):
        return self.out_dim

