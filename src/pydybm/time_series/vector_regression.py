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


__author__ = "Takayuki Osogami, Rudy Raymond"


from collections import deque
from six.moves import xrange, zip
from copy import deepcopy
from functools import partial

from .. import arraymath as amath
from ..time_series.time_series_model import StochasticTimeSeriesModel
from ..base.sgd import AdaGrad

DEBUG = False


def sigmoid(x):
    """
    Numerically stable implementation of sigmoid
    sigmoid(x) = 1 / (1 + exp(-x)
    Namely,
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        z = exp(x)
        return z / (1 + z)
    """
    return amath.exp(amath.minimum(0, x)) / (1 + amath.exp(-abs(x)))


class VectorRegression(StochasticTimeSeriesModel):

    """
    Vector regression for time series with unit variance

    Parameters
    ----------
    in_dim : int
        dimension of input time-series
    out_dim : int, optional
        dimension of target time-series
    order : int, optional
        order of the auto-regressive model
    SGD : Instance of SGD.SGD, optional
        Instance of a stochastic gradient method, default to AdaGrad()
    L1 : float, optional
        strength of L1 regularization
    L2 : float, optional
        strength of L2 regularization
    use_bias: boolean, optional
        whether to use bias parameters
    sigma : float, optional
        standard deviation of initial values of weight parameters
    random : arraymath.random, optional
        random number generator

    Attributes
    ----------
    len_fifo : int
        order
    in_dim : int
        in_dim
    out_dim : int
        out_dim
    variables : dict
        dictionary of model parameters
    variables["W"] : array, shape (len_fifo, in_dim, out_dim)
        variables["W"][l] corresponds to the weight from the input observed at
        time step t - l - 1 to the mean at time step t (current time).
    variables["b"] : array, shape (out_dim,)
        variables["b"] corresponds to the bias to out_pattern.
    fifo : deque
        FIFO queue storing len_fifo in_patterns, each in_pattern has shape (in_dim,).
    """

    def __init__(self, in_dim, out_dim=None, order=1, SGD=None, L1=0.,
                 L2=0., use_bias=True, sigma=0, random=None):
        if out_dim is None:
            out_dim = in_dim

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.len_fifo = order

        self.init_state()

        # y ~ N(mu, s)
        # mu = b + sum_n W[n] x[n]
        self.variables = dict()

        if self.len_fifo > 0 and self.in_dim > 0 and self.out_dim > 0:
            if sigma <= 0:
                self.variables["W"] \
                    = amath.zeros((self.len_fifo, self.in_dim, self.out_dim),
                                  dtype=float)
            else:
                if random is None:
                    random = amath.random.RandomState(0)
                self.variables["W"] \
                    = random.normal(0, sigma,
                                    (self.len_fifo, self.in_dim, self.out_dim))

        if use_bias and self.out_dim > 0:
            self.variables["b"] = amath.zeros((self.out_dim,), dtype=float)

        if SGD is None:
            SGD = AdaGrad()
        self.SGD = SGD.set_shape(self.variables)
        self.L2 = dict()
        self.L1 = dict()
        for key in self.variables:
            self.L1[key] = L1
            self.L2[key] = L2

        StochasticTimeSeriesModel.__init__(self)

    def init_state(self):
        """
        Initializing FIFO queues
        """
        self.fifo = amath.FIFO((max(0, self.len_fifo), self.in_dim))

    def _update_state(self, in_pattern):
        """
        Updating FIFO queue by appending in_pattern

        Parameters
        ----------
        in_pattern : array, shape (in_dim,)
            in_pattern to be appended to fifo.

        Returns
        -------
        popped_in_pattern : array, shape (in_dim,)
            in_pattern popped from fifo.
        """
        assert in_pattern.shape == (self.in_dim,), \
            "in_pattern must have shape (in_dim,):" + str(in_pattern.shape)

        if len(self.fifo) > 0:
            popped_in_pattern = self.fifo.push(in_pattern)
            return popped_in_pattern
        if len(self.fifo) == 0:
            return in_pattern

    def _get_delta(self, out_pattern, expected=None, weightLL=False):
        """
        Getting deltas, how much we change parameters by learning a given
        out_pattern

        Parameters
        ----------
        out_pattern : array, shape (out_dim,)
            out_pattern observed at the current time
        expected : array, shape (out_dim,), optional
            out_pattern expected by the current model.
            to be computed if not given.
        weightLL : boolean, optional
            whether to weight the delta by log-likelihood of out_pattern

        Returns
        -------
        dict
            dictionary of deltas with name of a variable as a key
        """
        assert out_pattern.shape == (self.out_dim,), \
            "out_pattern must have shape (out_dim,)"
        if expected is not None:
            assert expected.shape == (self.out_dim,), \
                "expected must have shape (out_dim,)"

        if self.SGD.get_learning_rate() == 0:
            return None

        gradient = self._get_gradient(out_pattern, expected)
        func_gradients = partial(
            self._func_gradients,
            order=self.len_fifo, in_dim=self.in_dim, out_dim=self.out_dim, fifo=self.fifo, out_pattern=out_pattern)
        self.SGD.update_state(gradient, self.variables, func_gradients)
        delta = self.SGD.get_delta()

        if weightLL:
            LL = self.get_LL(out_pattern)
            for key in delta:
                delta[key] *= LL
        return delta

    @staticmethod
    def _func_gradients(params, order, in_dim, out_dim, fifo, out_pattern):
        """
        Compute gradient with given output pattern.

        Parameters
        ----------
        params : Dictionary[str, amath.ndarray]
            Dictionary of parameters
        order : int
            Order of regression
        in_dim : int
            Dimensionality of input
        out_dim : int
            Dimensionality of output
        fifo : Iterable[amath.ndarray]
            FIFO queue containing past observations
        out_pattern : amath.ndarray
            Expected pattern of output

        Returns
        -------
        gradients : Dictionary[str, amath.ndarray]
            Dictionary of gradients
        """
        fifo_array = amath.array(fifo)
        L = order
        N = out_dim

        mu = amath.zeros((N, ))
        if "b" in params:
            mu[:] = params["b"].ravel()

        if L > 0:
            mu += amath.tensordot(fifo_array, params["W"], axes=2)

        if DEBUG:
            if "b" in params:
                mu_naive = deepcopy(params["b"]).ravel()
            else:
                mu_naive = amath.zeros((N,))
            for d in xrange(L):
                mu_naive = mu_naive + fifo[d].dot(params["W"][d])
            assert amath.allclose(mu, mu_naive), "ERROR: mu has a bug"
        expected = mu
        dx = out_pattern - expected  # just to avoid redundant computation

        gradient = dict()
        if "b" in params:
            gradient["b"] = dx
        if "W" in params:
            gradient["W"] = fifo_array[:, :, amath.newaxis] \
                            * dx[np.newaxis, amath.newaxis, :]
            if DEBUG:
                grad_W_naive = amath.array([fifo[d].reshape((in_dim, 1)) * dx
                                            for d in range(len(fifo))])
                assert amath.allclose(gradient["W"], grad_W_naive), \
                    "gradient[\"W\"] has a bug. \n{}\n{}\n{}".format(
                        gradient["W"], grad_W_naive, fifo)

        return gradient

    def _get_gradient(self, out_pattern, expected=None, applyL2=True):
        """
        Computing the gradient of log likelihood

        Parameters
        ----------
        out_pattern : array, shape (out_dim,)
            out_pattern observed at the current time
        expected : array, shape (out_dim,), optional
            out_pattern expected by the current model.
            to be computed if not given.
        applyL2 : boolean, optional
            if False, do not apply L2 regularization even if self.L2 > 0

        Returns
        -------
        dict
            dictionary of gradients with name of a variable as a key
        """
        assert out_pattern.shape == (self.out_dim,), \
            "out_pattern must have shape (out_dim,)"
        if expected is not None:
            assert expected.shape == (self.out_dim,), \
                "expected must have shape (out_dim,)"

        if expected is None:
            expected = self._get_mean()

        dx = out_pattern - expected  # just to avoid redundant computation

        gradient = dict()
        if "b" in self.variables:
            gradient["b"] = dx
        if "W" in self.variables:
            gradient["W"] = amath.op.mult_2d_1d_to_3d(self.fifo.to_array(), dx)
            if DEBUG:
                grad_W_naive = [self.fifo.to_array()[d].reshape((self.M, 1))
                                * dx for d in range(self.L)]
                assert amath.allclose(gradient["W"], grad_W_naive), \
                    "gradient[\"W\"] has a bug. \n{}\n{}\n{}".format(
                        gradient["W"], grad_W_naive, self.fifo)

        if applyL2:
            self.SGD.apply_L2_regularization(gradient, self.variables, self.L2)

        return gradient

    def _update_parameters(self, delta):
        """
        Updating parameters by delta

        Parameters
        ----------
        delta : dict
            dictionary of deltas for all variables
        """
        if delta is not None:
            self.SGD.update_with_L1_regularization(self.variables, delta, self.L1)

    def get_LL(self, out_pattern):
        """
        Computing the total LL of an out_pattern

        Parameters
        ----------
        out_pattern : array, length out_dim
            out_pattern observed at the current time

        Returns
        -------
        float
            total log likelihood of the out_pattern
        """
        if not out_pattern.shape == (self.out_dim,):
            raise ValueError("out_pattern must have shape (out_dim,)")

        mu = self._get_mean()
        LL = - 0.5 * (out_pattern-mu)**2 - 0.5 * amath.log(2 * amath.pi)
        return amath.sum(LL)

    def predict_next(self):
        """
        Predicting next out_pattern with the estimated mean

        Returns
        -------
        array, shape (out_dim, )
            prediction
        """
        return self._get_mean()

    def _get_mean(self):
        """
        Computing estimated mean

        Returns
        -------
        array, shape (out_dim,)
            estimated mean, or expected out_pattern in this case
        """
        return self._get_conditional_negative_energy()

    def _get_conditional_negative_energy(self):
        """
        Computing the conditional negative energy given fired

        Returns
        -------
        array, shape (out_dim,)
            fundamental output
        """
        mu = amath.zeros((self.out_dim, ))
        if "b" in self.variables:
            mu[:] = amath.array(self.variables["b"]).ravel()

        if "W" in self.variables:
            mu += amath.tensordot(self.fifo.to_array(),
                                  self.variables["W"],
                                  axes=2)

        if DEBUG:
            if "b" in self.variables:
                mu_naive = deepcopy(self.variables["b"]).ravel()
            else:
                mu_naive = amath.zeros((self.out_dim,))
            for d in xrange(self.len_fifo):
                mu_naive = mu_naive + self.fifo.to_array()[d].dot(
                    self.variables["W"][d])
            assert amath.allclose(mu, mu_naive), "ERROR: mu has a bug"

        return mu

    def set_learning_rate(self, rate):
        """
        Setting the learning rate

        Parameters
        ----------
        rate : float
            learning rate
        """
        self.SGD.set_learning_rate(rate)

    def _get_sample(self):
        """
        getting the next sample

        Returns
        -------
        array, shape (out_dim,)
            mu + n, where n is sampled from the standard normal distribution.
        """
        mu = self._get_mean()
        sample = self.random.normal(mu)
        return sample

    def get_sparsity(self, exclude=[]):
        """
        getting the sparsity of variables

        Parameters
        ----------
        exclude : list, optional
            list of the name of variables that should not be considered

        Returns
        -------
        float
            fraction of variables that are zeros
        """
        nnz = 0  # number of nonzero elements
        nz = 0  # number of zero elements
        for key in self.variables:
            if key in exclude:
                continue
            nnz += amath.sum(self.variables[key] != 0)
            nz += amath.sum(self.variables[key] == 0)
        sparsity = float(nz) / (nnz + nz)
        return sparsity

    def get_input_dimension(self):
        """
        Getting the dimension of the input sequence

        Returns
        -------
        in_dim : int
            dimension of input sequence
        """
        return self.in_dim

    def get_target_dimension(self):
        """
        Getting the dimension of the target sequence

        Returns
        -------
        out_dim : int
            dimension of target sequence
        """
        return self.out_dim


class MultiTargetVectorRegression(StochasticTimeSeriesModel):
    """

    Vector regression for multiple target time series with unit variance

    Parameters
    ----------
    in_dim : int
        dimension of input time-series
    out_dim : list of int, optional
        list of dimensions of target time-series
    order : int, optional
        order of the auto-regressive model
    SGDs : list of SGD
        list of objects of stochastic gradient method
    order : int, optional
        order of the auto-regressive model
    L1 : float, optional
        strength of L1 regularization
    L2 : float, optional
        strength of L2 regularization
    use_bias: boolean, optional
        whether to use bias parameters

    Attributes
    ----------
    layers : list of VectorRegression
    """

    def __init__(self, in_dim, out_dims, SGDs, order=1, L1=0., L2=0.,
                 use_bias=True):

        self.layers = [VectorRegression(in_dim, out_dim, order, SGD, L1, L2,
                                        use_bias)
                       for (out_dim, SGD) in zip(out_dims, SGDs)]

        # Only layer 0 has the internal states, which are shared among
        # all layers
        for i in xrange(1, len(self.layers)):
            self.layers[i].fifo = self.layers[0].fifo

        StochasticTimeSeriesModel.__init__(self)

    def init_state(self):
        """
        Initializing FIFO queues
        """
        self.layers[0].init_state()

    def _update_state(self, in_pattern):
        """
        Updating FIFO queue by appending in_pattern

        Parameters
        ----------
        in_pattern : array, shape (in_dim,)
            in_pattern to be appended to fifo.

        Returns
        -------
        popped_in_pattern : array, shape (in_dim,)
            in_pattern popped from fifo.
        """
        return self.layers[0]._update_state(in_pattern)

    def _get_delta(self, out_patterns, expecteds=None, weightLLs=None):
        """
        Getting deltas, how much we change parameters by learning a given
        out_pattern

        Parameters
        ----------
        out_patterns : list, length len(layers)
            out_patterns[l] : array, shape (layers[l].out_dim,)
            out_pattern of layer l observed at the current time
        expecteds : list
            expecteds[l] : array, shape (layers[l].out_dim,), optional
            out_pattern of layer l expected by the current model.
            to be computed if not given.
        weightLLs : list, optional
            weightLLs[l] : whether to weight delta by log likelihood of
            out_patterns[l]

        Returns
        -------
        list of dict
            list of dictionary of deltas with name of a variable as a key
        """
        assert len(out_patterns) == len(self.layers), \
            "length of out_patterns must match number of layers"

        if expecteds is None:
            expecteds = [None] * len(self.layers)

        if weightLLs is None:
            weightLLs = [False] * len(self.layers)

        assert len(expecteds) == len(self.layers), \
            "length of expected must match number of layers"

        return [layer._get_delta(out_pattern, expected, weightLL)
                for (layer, out_pattern, expected, weightLL)
                in zip(self.layers, out_patterns, expecteds, weightLLs)]

    def _get_gradient(self, out_patterns, expecteds=None):
        """
        Computing the gradient of log likelihood

        Parameters
        ----------
        out_patterns : list, length len(layers)
            out_patterns[l] : array, shape (layers[l].out_dim,)
            out_pattern of layer l observed at the current time
        expecteds : list
            expecteds[l] : array, shape (layers[l].out_dim,), optional
            out_pattern of layer l expected by the current model.
            to be computed if not given.

        Returns
        -------
        list of dict
            list of dictionary of gradients with name of a variable as a key
        """
        assert len(out_patterns) == len(self.layers), \
            "length of out_patterns must match number of layers"
        if expecteds is not None:
            assert len(expecteds) == len(self.laners), \
                "length of expected must match number of layers"

        return [layer._get_gradient(out_pattern, expected)
                for (layer, out_pattern, expected)
                in zip(self.layers, out_patterns, expecteds)]

    def _get_gradient_for_layer(self, out_pattern, layer, expected):
        """
        Computing the gradient of log likelihood

        Parameters
        ----------
        out_pattern : array, shape (layers[layer].out_dim,)
            out_pattern of the layer observed at the current time
        layer : int
            index of the layer
        expected : array, shape (layers[layer].out_dim)
            expected pattern of the layer

        Returns
        -------
        list of dict
            list of dictionary of gradients with name of a variable as a key
        """

        return self.layers[layer]._get_gradient(out_pattern, expected)

    def _update_parameters(self, deltas):
        """
        Updating parameters by deltas

        Parameters
        ----------
        delta : list of dict
            list of dictionary of deltas for all variables
        """
        assert len(deltas) == len(self.layers), \
            "length of deltas must match number of layers"

        if deltas is not None:
            for (layer, delta) in zip(self.layers, deltas):
                layer._update_parameters(delta)

    def get_LL(self, out_patterns):
        """
        Computing the total LL of an out_pattern

        Parameters
        ----------
        out_patterns : list, length len(layers)
            out_patterns[l] : array, shape (layers[l].out_dim,)
            out_pattern of layer l observed at the current time

        Returns
        -------
        list of float
            list of total log likelihood of the out_pattern
        """
        if not len(out_patterns) == len(self.layers):
            raise ValueError("length of out_patterns must match number of "
                             "layers")

        return [layer.get_LL(out_pattern)
                for (layer, out_pattern) in zip(self.layers, out_patterns)]

    def predict_next(self):
        """
        Predicting next out_pattern with the estimated mean

        Returns
        -------
        list of array, length len(layers)
            list of prediction
        """
        return [layer._get_mean() for layer in self.layers]

    def _get_mean(self):
        """
        Computing estimated mean

        Returns
        -------
        mu : list of array, length len(layers)
            list of estimated mean
        """
        return [layer._get_mean() for layer in self.layers]

    def _get_conditional_negative_energy(self):
        """
        Computing the conditional negative energy given fired

        Returns
        -------
        list of array, length len(layers)
            list of fundamental output
        """
        return [layer._get_conditional_negative_energy()
                for layer in self.layers]

    def set_learning_rate(self, rates):
        """
        Setting the learning rate

        Parameters
        ----------
        rate : list of float, length len(layers)
            list of learning rate
        """

        for (layer, rate) in zip(self.layers, rates):
            layer.set_learning_rate(rate)

    def get_sparsity(self, excludes=[]):
        """
        getting the sparsity of variables

        Parameters
        ----------
        excludes : list, optional
            list of the name of variables that should not be considered

        Returns
        -------
        list of float
            list of fraction of variables that are zeros
        """
        return [layer.get_sparsity(excludes) for layer in self.layers]

    def get_input_dimension(self):
        """
        Getting the dimension of input sequence

        Returns
        -------
        in_dim : int
            dimension of input sequence
        """
        return self.layers[0].get_input_dimension()

    def get_target_dimension(self):
        """
        Getting the dimension of target sequence

        Returns
        -------
        out_dim : int
            dimension of target sequence
        """
        return [layer.get_target_dimension() for layer in self.layers]

    def _get_sample(self):
        """
        Getting samples from each layer

        Returns
        -------
        list of arrays, length len(layers)
        """
        return [layer._get_sample() for layer in self.layers]


class VectorRegressionWithVariance(VectorRegression):

    """
    Vector regression for time series.
    The variance is also a model parameter to be estimated.

    Parameters
    ----------
    in_dim : int
        dimension of input time-series
    out_dim : int, optional
        dimension of target time-series
    order : int, optional
        order of the auto-regressive model
    SGD : object of SGD.SGD, optional
        object of a stochastic gradient method, default to AdaGrad()
    L1 : float, optional
        strength of L1 regularization
    L2 : float, optional
        strength of L2 regularization

    Attributes
    ----------
    len_fifo : int
        order
    in_dim : int
        in_dim
    out_dim : int
        out_dim
    variables : dict
        dictionary of model parameters
    variables["W"] : array, shape (len_fifo, in_dim, out_dim)
        variables["W"][l] corresponds to the weight from the input observed at
        time step t - l - 1 to the mean at time step t (current time).
    variables["b"] : array, shape (1, out_dim)
        variables["b"] corresponds to the bias to out_pattern.
    variables["s"] : array, shape (1, out_dim)
        variables["s"][n] corresponds to the standard deviation of
        out_pattern[n] (or scale parameter in other words)
    fifo : deque
        FIFO queue storing len_fifo in_patterns, each in_pattern has shape (in_dim,).
    """

    def __init__(self, in_dim, out_dim=None, order=1, SGD=None, L1=0.,
                 L2=0.):
        VectorRegression.__init__(self, in_dim, out_dim, order)
        self.variables["s"] = amath.ones((1, self.out_dim), dtype=float)
        if SGD is None:
            SGD = AdaGrad()
        self.SGD = SGD.set_shape(self.variables)
        for key in self.variables:
            self.L1[key] = L1
            self.L2[key] = L2
        self.L1["s"] = 0.
        self.L2["s"] = 0.

    def _get_gradient(self, out_pattern, expected=None):
        """
        Computing the gradient of log likelihood

        Parameters
        ----------
        out_pattern : array, length out_dim
            out_pattern observed at the current time
        expected : array, length out_dim, optional
            out_pattern expected by the current model.
            to be computed if not given.

        Returns
        -------
        dict
            dictionary of gradients with name of a variable as a key
        """
        assert out_pattern.shape == (self.out_dim,), \
            "out_pattern must have shape (out_dim,)"

        if expected is None:
            expected = self._get_mean()

        gradient = VectorRegression._get_gradient(self, out_pattern,
                                                  expected, False)

        x = out_pattern.reshape((1, self.out_dim))

        NATURAL_GRADIENT = True

        if NATURAL_GRADIENT:
            dxdx = (x - expected)**2
            gradient["s"] \
                = 0.5 * dxdx / self.variables["s"] \
                - 0.5 * self.variables["s"]
            # standard deviation "s" can be replaced with variance or precision
            # gradient["variance"] = dxdx - self.variables["variance"]
            # gradient["precision"] = self.variables["precision"] \
            #    - dxdx * self.variables["precision"]**2
        else:
            if "b" in self.variables:
                gradient["b"] = amath.op.divide_by_pow(gradient["b"],
                                                       self.variables["s"],
                                                       2)
            gradient["s"] = amath.op.vecreg_gradient_s(self.variables["s"],
                                                       out_pattern,
                                                       expected)
            if self.len_fifo > 0:
                gradient["W"] = amath.op.divide_3d_by_1d_pow(
                    gradient["W"], self.variables["s"], 2)

        self.SGD.apply_L2_regularization(gradient, self.variables, self.L2)

        return gradient

    def _update_parameters(self, delta):
        """
        Updating parameters by delta

        Parameters
        ----------
        delta : dict
            dictionary of arraymath array of amount of changes to the variables
        """
        if delta is not None:
            VectorRegression._update_parameters(self, delta)

    def get_LL(self, out_pattern):
        """
        Computing the total LL of an out_pattern

        Parameters
        ----------
        out_pattern : array, length out_dim
            out_pattern observed at the current time

        Returns
        -------
        float
            total log likelihood
        """
        if not out_pattern.shape == (self.out_dim,):
            raise ValueError("out_pattern must have shape (out_dim,)")

        x = out_pattern.reshape((1, self.out_dim))
        mu = self._get_mean().reshape((1, self.out_dim))
        s = self.variables["s"]
        LL = - 0.5 * (x - mu)**2 / s**2 - 0.5 * amath.log(2 * s**2 * amath.pi)
        return amath.sum(LL)

    def _get_sample(self):
        """
        Getting the next sample

        Returns
        -------
        array, shape (out_dim)
            mu + n, where n ~ N(0, variables["s"] ** 2)
        """
        mu = self._get_mean().reshape((1, self.out_dim))
        sigma = self.variables["s"]
        sample = self.random.normal(mu, sigma)
        sample = sample.reshape(self.out_dim)
        return sample


class VectorLogisticRegression(VectorRegression):

    """
    Vector logistic regression for time series.
    out_pattern ~ Bern(sigmoid(mu)), where mu is updated according to the same
    rule as VectorRegression.

    Parameters
    ----------
    in_dim : int
        dimension of input time-series
    out_dim : int, optional
        dimension of target time-series
    order : int, optional
        order of the auto-regressive model
    SGD : object of SGD.SGD, optional
        object of a stochastic gradient method
    L1 : float, optional
        strength of L1 regularization
    L2 : float, optional
        strength of L2 regularization

    Attributes
    ----------
    len_fifo : int
        order
    in_dim : int
        in_dim
    out_dim : int
        out_dim
    variables : dict
        dictionary of model parameters
    variables["W"] : array, shape (len_fifo, in_dim, out_dim)
        variables["W"][l] corresponds to the weight from the input observed at
        time step t - l - 1 to mu at time step t (current time).
    variables["b"] : array, shape (out_dim,)
        variables["b"] corresponds to the bias to mu.
    fifo : deque
        FIFO queue storing len_fifo in_patterns, each in_pattern has shape (in_dim,).
    """

    def predict_next(self):
        """
        Predicting next out_pattern with the estimated mean
        (equivalently, firing probabilities)

        Returns
        -------
        array, shape (out_dim,)
            prediction
        """
        return self._get_mean()

    def get_LL(self, out_pattern):
        """
        Computing the total LL of an out_pattern

        Parameters
        ----------
        out_pattern : array, length out_dim
            out_pattern observed at the current time

        Returns
        -------
        float
            total log likelihood
        """
        if not out_pattern.shape == (self.out_dim,):
            raise ValueError("out_pattern must have shape (out_dim,)")

        mu = self._get_conditional_negative_energy()
        LL = -mu * out_pattern - amath.log(1. + amath.exp(-mu))
        return amath.sum(LL)

    def _get_mean(self):
        """
        Computing estimated mean

        Returns
        -------
        array, shape (out_dim,)
            estimated mean
        """
        mu = self._get_conditional_negative_energy()
        return sigmoid(mu)

    def _get_sample(self):
        """
        Getting the next sample

        Returns
        -------
        array, shape (out_dim,)
            sampled from Bernouilli distributions with the estimated means
        """
        p = self._get_mean()
        u = self.random.random_sample(p.shape)
        sample = u < p
        return sample


class VectorRegressionWithHidden(VectorRegression):
    """
    Vector regression with one hidden layer

    Parameters
    ----------
    in_dim : int
        dimension of input time-series
    out_dim : int, optional
        dimension of target time-series
    dim_hidden : int, optional
        dimension of a hidden layer
    order : int, optional
        order of the auto-regressive model. order >= 1.
    SGD : object of SGD.SGD, optional
        object of a stochastic gradient method, default to AdaGrad()
    L1 : float, optional
        strength of L1 regularization
    L2 : float, optional
        strength of L2 regularization
    use_bias: boolean, optional
        whether to use bias parameters
    sigma : float, optional
        standard deviation of initial values of weight parameters
    random : arraymath.random, optional
        random number generator

    Attributes
    ----------
    len_fifo : int
        order
    in_dim : int
        in_dim
    out_dim : int
        out_dim
    dim_hidden : int
        dimension of a hidden layer
    variables : dict
        dictionary of model parameters
    variables["W"] : array, shape (len_fifo, in_dim, out_dim)
        variables["W"][l] corresponds to the weight from the input observed at
        time step t - l - 1 to the mean at time step t (current time).
    variables["U"] : array, shape (len_fifo, in_dim, dim_hidden)
        variables["U"][l] corresponds to the weight from the input observed at
        time step t - l - 1 to the hidden layer at time step t - 1.
    variables["V"] : array, shape (dim_hidden, out_dim)
        variables["V"] corresponds to the weight from the hidden variables at
        time step t - 1 to the output layer at time step t (current time).
    variables["b"] : array, shape (out_dim,)
        variables["b"] corresponds to the bias to out_pattern.
    variables["b_h"] : array, shape (dim_hidden,)
        variables["b_h"] corresponds to the bias to the hidden layer.
    fifo : deque
        FIFO queue storing len_fifo in_patterns, each in_pattern has shape (in_dim,).
    """

    def __init__(self, in_dim, out_dim=None, dim_hidden=1, order=1,
                 SGD=None, L1=0., L2=0., use_bias=True, sigma=0, random=None):
        if not order >= 1:
            raise ValueError("order must satisfy `order >= 1`.")
        super(VectorRegressionWithHidden, self).__init__(in_dim, out_dim,
                                                         order, SGD, L1, L2,
                                                         use_bias, sigma)
        self.dim_hidden = dim_hidden

        if sigma <= 0:
            self.variables["b_h"] = amath.zeros((self.dim_hidden,),
                                                dtype=float)
            self.variables["U"] = amath.zeros(
                (self.len_fifo, self.in_dim, self.dim_hidden),
                dtype=float)
            self.variables["V"] = amath.zeros((self.dim_hidden, self.out_dim),
                                              dtype=float)
        else:
            if random is None:
                random = amath.random.RandomState(0)
            self.variables["b_h"] = random.normal(0, sigma, (self.dim_hidden,))
            self.variables["U"] \
                = random.normal(0, sigma, (self.len_fifo, self.in_dim, self.dim_hidden))
            self.variables["V"] \
                = random.normal(0, sigma, (self.dim_hidden, self.out_dim))
        if SGD is None:
            SGD = AdaGrad()
        self.SGD = SGD.set_shape(self.variables)
        self.L2 = dict()
        self.L1 = dict()
        for key in self.variables:
            self.L1[key] = L1
            self.L2[key] = L2

    def get_LL(self, out_pattern):
        """ get the lower-bound of the log-likelihood

        Parameters
        ----------
        out_pattern : array, shape (out_dim,)
            out_pattern observed at the current time

        Returns
        -------
        float
            the lower-bound of the log-likelihood
        """
        return self._get_obj(out_pattern)

    def _get_obj(self, out_pattern):
        """ compute the lower-bound of the log-likelihood.

        Parameters
        ----------
        out_pattern : array, shape (out_dim,)
            out_pattern observed at the current time

        Returns
        -------
        float
            the lower-bound of the log-likelihood
        """
        x_tilde = self._get_x_tilde(out_pattern)
        sig_u_tilde = amath.exp(self._get_u_tilde(log_sigmoid=True))
        V_times_sig_u_tilde = sig_u_tilde.dot(self.variables["V"])

        obj = - 0.5 * self.out_dim * amath.log(2.0 * amath.pi) \
              - 0.5 * amath.inner(x_tilde, x_tilde)
        obj = obj + amath.inner(x_tilde, V_times_sig_u_tilde)
        obj = obj - 0.5 * amath.inner(V_times_sig_u_tilde, V_times_sig_u_tilde)
        obj = obj - 0.5 * amath.inner(sig_u_tilde,
                                      (1.0 - sig_u_tilde)
                                      * amath.diag(self.variables["V"].dot(
                                          self.variables["V"].transpose())))
        return obj

    def _get_gradient(self, out_pattern, expected=None):
        """ compute the gradient of the lower boudn with respect to each
        parameter

        Parameters
        ----------
        out_pattern : array, shape (out_dim,)
            out_pattern observed at the current time
        expected : array, shape (out_dim,)
            expected pattern

        Returns
        -------
        dict
            directory of gradients with name of a variable as a key
        """
        assert out_pattern.shape == (self.out_dim,), \
            "out_pattern must have shape ({},), not {}.".format(
                self.out_dim, out_pattern.shape)

        x_tilde = self._get_x_tilde(out_pattern)
        u_tilde = self._get_u_tilde(log_sigmoid=False)
        sig_u_tilde = amath.exp(self._get_u_tilde(log_sigmoid=True))
        V_times_sig_u_tilde = sig_u_tilde.dot(self.variables["V"])
        mu = self._get_mean()

        gradient = dict()
        gradient["b"] = out_pattern - mu
        gradient["W"] \
            = self.fifo.to_array()[:, :, amath.newaxis] \
            * gradient["b"][amath.newaxis, amath.newaxis, :]
        if DEBUG:
            grad_W_naive = amath.array([amath.outer(self.fifo.to_array()[d],
                                                    gradient["b"])
                                        for d in xrange(self.len_fifo)])
            assert amath.allclose(gradient["W"], grad_W_naive), \
                "gradient[\"W\"] has a bug."

        gradient["V"] \
            = amath.outer(sig_u_tilde, x_tilde) \
            - (amath.outer(sig_u_tilde, sig_u_tilde)
               + amath.diag(sig_u_tilde
                            * (1.0 - sig_u_tilde))).dot(self.variables["V"])
        grad_u_tilde \
            = sig_u_tilde * (1.0 - sig_u_tilde) \
            * (self.variables["V"].dot(x_tilde - sig_u_tilde.dot(self.variables["V"]))
               + amath.diag(self.variables["V"].dot(
                   self.variables["V"].transpose()))
               * (sig_u_tilde - 0.5))
        gradient["b_h"] = grad_u_tilde
        gradient["U"] \
            = self.fifo.to_array()[:, :, amath.newaxis] \
            * grad_u_tilde[amath.newaxis, amath.newaxis, :]
        if DEBUG:
            grad_U_naive = amath.array([amath.outer(self.fifo.to_array()[d],
                                                    grad_u_tilde)
                                        for d in xrange(self.len_fifo)])
            assert amath.allclose(gradient["U"], grad_U_naive), \
                "gradient[\"U\"] has a bug."
        return gradient

    def init_state(self):
        """ init fifo
        """
        super(VectorRegressionWithHidden, self).init_state()

    def _update_state(self, in_pattern):
        """
        Updating FIFO queue by appending in_pattern

        Parameters
        ----------
        in_pattern : array, shape (in_dim,)
            in_pattern to be appended to fifo.

        Returns
        -------
        popped_in_pattern : array, shape (in_dim,)
            in_pattern popped from fifo.
        """
        popped_in_pattern \
            = super(VectorRegressionWithHidden, self)._update_state(in_pattern)
        return popped_in_pattern

    def _get_conditional_negative_energy(self):
        """ compute mu, which can be used for prediction of the next pattern.

        Returns
        -------
        array, shape (out_dim,)
            mu, the mean of the output layer
        """
        mu = amath.zeros((self.out_dim, ))
        if "b" in self.variables:
            mu[:] = self.variables["b"]

        mu += amath.tensordot(self.fifo.to_array(), self.variables["W"], axes=2)

        if DEBUG:
            if "b" in self.variables:
                mu_naive = deepcopy(self.variables["b"])
            else:
                mu_naive = amath.zeros((self.out_dim,))
            for d in xrange(self.len_fifo):
                mu_naive = mu_naive + self.fifo.to_array()[d].dot(
                    self.variables["W"][d])
            assert amath.allclose(mu, mu_naive), "ERROR: mu has a bug"
        sig_u_tilde = amath.exp(self._get_u_tilde(log_sigmoid=True))
        mu = mu + sig_u_tilde.dot(self.variables["V"])
        return mu

    def _get_u_tilde(self, log_sigmoid=False):
        """ Compute u_tilde, which determines the energy of the hidden
        variables.  The energy is defined as the inner product of hidden
        variables and u_tilde.

        Parameters
        ----------
        log_sigmoid : bool
            if True, return log_sigmoid(u_tilde)

        Returns
        -------
        array, shape (dim_hidden,)
            u_tilde
        """
        u_tilde = amath.zeros(self.dim_hidden)
        u_tilde += self.variables["b_h"]
        u_tilde += amath.tensordot(self.fifo.to_array(),
                                   self.variables["U"], axes=2)

        if DEBUG:
            u_tilde_naive = amath.zeros(self.dim_hidden)
            u_tilde_naive = u_tilde_naive + self.variables["b_h"]
            for d in xrange(self.len_fifo):
                u_tilde_naive = u_tilde_naive \
                    + self.fifo.to_array()[d].dot(self.variables["U"][d])
            assert amath.allclose(u_tilde, u_tilde_naive), \
                "ERROR: u_tilde has a bug."

        if log_sigmoid:
            ll = amath.log_logistic(u_tilde)
            u_tilde = amath.array(ll).reshape(u_tilde.shape)
        return u_tilde

    def _get_x_tilde(self, out_pattern):
        """ Compute x_tilde, the difference between out_pattern and mu
        (without hidden units)

        Parameters
        ----------
        out_pattern : array, shape (out_dim,)

        Returns
        -------
        array, shape (out_dim,)
            x[t] - b - sum_d W[d] x[t-d]
        """
        assert out_pattern.shape == (self.out_dim,), \
            "out_pattern must have shape ({},), not {}.".format(
                self.out_dim, out_pattern.shape)
        x_tilde = amath.zeros(self.out_dim)
        x_tilde += out_pattern
        if "b" in self.variables:
            x_tilde -= self.variables["b"]

        if DEBUG:
            x_tilde_naive = deepcopy(x_tilde)
            for d in xrange(self.len_fifo):
                x_tilde_naive = x_tilde_naive \
                    - self.fifo.to_array()[d].dot(self.variables["W"][d])

        x_tilde -= amath.tensordot(self.fifo.to_array(),
                                   self.variables["W"], axes=2)

        if DEBUG:
            assert amath.allclose(x_tilde, x_tilde_naive), \
                "ERROR: x_tilde has a bug."
        return x_tilde

    def get_sparsity(self, exclude=[]):
        # TODO: implement get_sparsity
        raise NotImplementedError("get_sparsity not implemented for VectorRegressionWithHidden")
