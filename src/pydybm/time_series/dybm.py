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

"""Implementation of DyBMs.

.. seealso:: Takayuki Osogami and Makoto Otsuka, "Seven neurons \
memorizing sequences of alphabetical images via spike-timing dependent \
plasticity," Scientific Repeports, 5, 14149; doi: 10.1038/srep14149 \
(2015).  http://www.nature.com/articles/srep14149

.. seealso:: Takayuki Osogami and Makoto Otsuka, Learning dynamic \
Boltzmann machines with spike-timing dependent plasticity, Technical \
Report RT0967, IBM Research, 2015.  https://arxiv.org/abs/1509.08634
"""

__author__ = "Takayuki Osogami, Rudy Raymond"


import numpy as np
from six.moves import xrange, zip
from copy import deepcopy
from itertools import product
from .. import arraymath as amath
from ..base.generator import ListGenerator, ElementGenerator, SequenceGenerator
from ..time_series.time_series_model import StochasticTimeSeriesModel
from ..time_series.vector_regression import VectorRegression, MultiTargetVectorRegression, \
    VectorRegressionWithHidden, sigmoid
from ..base.sgd import AdaGrad
from ..base.metrics import RMSE, baseline_RMSE

DEBUG = False


class LinearDyBM(VectorRegression):

    """LinearDyBM is a minimal DyBM for real-valued time-series.  It
    extends a vector auto-regressive (VAR) model by incorporating
    eligibility traces (and an Echo State Network or ESN).
    Specifically, a pattern, :math:`\mathbf{x}^{[t]}`, at time
    :math:`t` is predicted with

    .. math::

        \mathbf{b}
        + \sum_{\ell=1}^{L} \mathbf{W}^{[\ell]} \, \mathbf{x}^{[t-\ell]}
        + \sum_{k=1}^{K} \mathbf{V}^{[k]} \, \mathbf{e}^{[k]}
        + \\Phi^{[t-1]}

    Here, :math:`\mathbf{x}^{[t-\ell]}` is the pattern at time
    :math:`t-\ell`, and :math:`\mathbf{e}^{[k]}` is a vector of
    eligibility traces, which are updated as follows after receiving a
    pattern at each time :math:`t`:

    .. math::

        \mathbf{e}^{[k]} \leftarrow \lambda^{[k]} \, \mathbf{e}^{[k]}
        + \mathbf{x}^{[t]}

    where :math:`\lambda^{[k]}` is the :math:`k`-th decay rate.

    Optionally, LinearDyBM can also take into account the
    supplementary bias, :math:`\\Phi^{[t]} = \mathbf{A} \\Psi^{[t]}`,
    created by an ESN from the (non-linear) features :math:`\\Psi^{[t]}`,
    as follows (Notice that :math:`\mathbf{A}` is learned by the ESN):

    .. math::

        \\Psi^{[t]}
        = (1-\\rho) \, \\Psi^{[t-1]}
        + \\rho \, \mathcal{F}(\mathbf{W}_{rec} \, \\Psi^{[t-1]}
                               + \mathbf{W}_{in} \, \mathbf{x}^{[t]})

    where :math:`\mathbf{W}_{rec}` is the matrix of the internal
    weight in the ESN, :math:`\mathbf{W}_{in}` is the matrix of
    the weight from the input to the ESN, :math:`\rho` is a leak parameter,
    and :math:`\mathcal{F}` is a non-linear function (specifically, hyperbolic tangent).  These
    :math:`\mathbf{W}_{rec}` and :math:`\mathbf{W}_{in}` are randomly
    initialized and fixed throughout learning.

    LinearDyBM learns the values of :math:`\mathbf{V}^{[\cdot]}`,
    :math:`\mathbf{W}^{[\cdot]}` from given
    training time-series.  The ESN is used differently in Dasgupta &
    Osogami (2017).  See RNNGaussianDyBM.py for the implementation
    that follows Dasgupta & Osogami (2017).  The LinearDyBM without an
    ESN closely follows Osogami (2016).

    .. seealso:: Takayuki Osogami, "Learning binary or real-valued \
    time-series via spike-timing dependent plasticity," presented at \
    Computing with Spikes NIPS 2016 Workshop, Barcelona, Spain, December \
    2016.  https://arxiv.org/abs/1612.04897

    .. seealso:: Sakyasingha Dasgupta and Takayuki Osogami, "Nonlinear \
    Dynamic Boltzmann Machines for Time-series Prediction," in \
    Proceedings of the 31st AAAI Conference on Artificial Intelligence \
    (AAAI-17), pages 1833-1839, San Francisco, CA, January 2017. \
    http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14350/14343

    LinearDyBM allows the target time-series to be different from the input
    time-series.  Namely, LinearDyBM can be trained to predict a target
    pattern, :math:`\mathbf{y}^{[t]}`, at time :math:`t` from input time-series
    :math:`\mathbf{x}^{[:t]}` (i.e., `\mathbf{x}^{[0]`, ..., `\mathbf{x}^{[t-1]`) \
    with

    .. math::

        \mathbf{b}
        + \sum_{\ell=1}^{L} \mathbf{W}^{[\ell]} \, \mathbf{x}^{[t-\ell]}
        + \sum_{k=1}^{K} \mathbf{V}^{[k]} \, \mathbf{e}^{[k]}
        + \\Phi^{[t-1]}

    Note that $\mathbf{x}^{[t]}$ is not used to predict $\mathbf{y}^{[t]}$.

    Parameters
    ----------
    in_dim : int
        Dimension of input time-series
    out_dim : int, optional
        Dimension of target time-series
    delay : int, optional
        length of the FIFO queue plus 1
    decay_rates : list, optional
        Decay rates of eligibility traces
    SGD : instance of SGD.SGD, optional
        Instance of a stochastic gradient method
    L1 : float, optional
        Strength of L1 regularization
    L2 : float, optional
        Strength of L2 regularization
    use_bias : boolean, optional
        Whether to use bias parameters
    sigma : float, optional
        Standard deviation of initial values of weight parameters
    insert_to_etrace : str, {"w_delay", "wo_delay"},  optional
        "w_delay" : Insert pattern observed d-1 time steps ago into
        eligibility traces.
        "wo_delay" : Insert the latest pattern into eligibility traces
    esn : ESN, optional
        Echo state network
    random : arraymath.random
        random number generator

    Attributes
    ----------
    decay_rates : array, shape (n_etrace, 1)
        Decay rates of eligibility traces
    e_trace : array, shape (n_etrace, in_dim)
        e_trace[k, :] corresponds to the k-th eligibility trace.
    esn : ESN
        esn
    fifo : deque
        FIFO queue storing L in_patterns, each in_pattern has shape (in_dim,).
    insert_to_etrace : str
        insert_to_etrace
    n_etrace : int
        The number of eligibility traces
    len_fifo : int
        The length of FIFO queues (delay - 1)
    L1 : dict
        Dictionary of the strength of L1 regularization
    L1[x] : float
        Strength of L1 regularization for variable x for x in ["b","V","W"]
    L2 : dict
        Dictionary of the strength of L2 regularization
    L2[x] : float
        Strength of L2 regularization for variable x for x in ["b","V","W"]
    in_dim : int
        in_dim
    out_dim : int
        out_dim
    SGD : SGD
        Optimizer used in the stochastic gradient method
    variables : dict
        Dictionary of model parameters
    variables["W"] : array, shape (len_fifo, in_dim, out_dim)
        variables["W"][l] corresponds to the weight
        from the input observed at time step t - l - 1
        to the mean at time step t (current time).
    variables["b"] : array, shape (out_dim,)
        variables["b"] corresponds to the bias to out_pattern.
    variables["V"] : array, shape (n_etrace, in_dim, out_dim)
        variables["V"][k] corresponds to the weight
        from the k-th eligibility trace to the mean.
    """

    def __init__(self, in_dim, out_dim=None, delay=2, decay_rates=[0.5],
                 SGD=None, L1=0, L2=0, use_bias=True, sigma=0,
                 insert_to_etrace="wo_delay", esn=None, random=None):

        if insert_to_etrace not in ["w_delay", "wo_delay"]:
            raise ValueError("insert_to_etrace should be either 'w_delay' "
                             "or 'wo_delay'.")

        self.n_etrace = len(decay_rates)
        self.decay_rates = amath.array(decay_rates).reshape((self.n_etrace, 1))
        order = delay - 1

        # Echo state network
        self.esn = esn
        if SGD is None:
            SGD = AdaGrad()

        if random is None:
            random = amath.random.RandomState(0)

        VectorRegression.__init__(self, in_dim, out_dim, order, SGD, L1, L2,
                                  use_bias, sigma, random)
        if self.n_etrace > 0 and self.in_dim > 0 and self.out_dim > 0:
            if sigma <= 0:
                self.variables["V"] \
                    = amath.zeros((self.n_etrace, self.in_dim, self.out_dim),
                                  dtype=float)
            else:
                self.variables["V"] \
                    = random.normal(0, sigma,
                                    (self.n_etrace, self.in_dim, self.out_dim))

        if SGD is None:
            SGD = AdaGrad()
        self.SGD = SGD.set_shape(self.variables)  # resetting SGD
        self.L1["V"] = L1
        self.L2["V"] = L2
        self.insert_to_etrace = insert_to_etrace

    def init_state(self):
        """Initializing FIFO queue and eligibility traces
        """
        VectorRegression.init_state(self)
        # eligibility trace
        self.e_trace = amath.zeros((self.n_etrace, self.in_dim))

        # Echo state network
        if self.esn is not None:
            self.esn.init_state()

    def _update_state(self, in_pattern):
        """Updating FIFO queues and eligibility traces by appending in_pattern

        Parameters
        ----------
        in_pattern : array, shape (in_dim,)
            in_pattern to be appended to fifo.
        """
        assert in_pattern.shape == (self.in_dim,), "in_pattern must have shape (in_dim,)"

        popped_in_pattern = VectorRegression._update_state(self, in_pattern)
        if self.insert_to_etrace == "wo_delay" and self.in_dim > 0:
            self.e_trace = amath.op.update_e_trace(self.e_trace,
                                                   self.decay_rates,
                                                   in_pattern)
        elif self.insert_to_etrace == "w_delay" and self.in_dim > 0:
            self.e_trace = amath.op.update_e_trace(self.e_trace,
                                                   self.decay_rates,
                                                   popped_in_pattern)
        elif self.in_dim > 0:
            raise NotImplementedError("_update_state not implemented for ",
                                      self.insert_to_etrace, self.in_dim)
        else:
            # no need to do anything when in_dim == 0
            pass

        # Echo state network
        if self.esn is not None:
            self.esn._update_state(in_pattern)

    def _get_gradient(self, out_pattern, expected=None):
        """Computing the gradient of log likelihood

        Parameters
        ----------
        out_pattern : array, shape (out_dim,)
            out_pattern observed at the current time
        expected : array, shape (out_dim,), optional
            out_pattern expected by the current model.
            To be computed if not given.

        Returns
        -------
        dict
            dictionary of gradients with name of a variable as a key
        """

        assert out_pattern.shape == (self.out_dim,), \
            "out_pattern must have shape (out_dim,)"

        if expected is not None:
            assert expected.shape == (self.out_dim,), "expected must have shape (out_dim,)"

        if expected is None:
            expected = self._get_mean().reshape((self.out_dim,))

        gradient = VectorRegression._get_gradient(self, out_pattern, expected)

        if "V" in self.variables:
            # TODO:
            # dx in the following has been computed
            # in VectorRegression._get_gradient
            dx = out_pattern - expected
            gradient["V"] = amath.op.mult_2d_1d_to_3d(self.e_trace, dx)
            if DEBUG:
                grad_V_naive \
                    = np.array([self.e_trace[k, :].reshape((self.in_dim, 1)) * dx
                                for k in range(self.n_etrace)])
                assert np.allclose(grad_V_naive, gradient["V"]), \
                    "gradient[\"V\"] has a bug."

        self.SGD.apply_L2_regularization(gradient, self.variables, self.L2)

        return gradient

    def learn_one_step(self, out_pattern):
        """Learning a pattern and updating parameters

        Parameters
        ----------
        out_pattern : array, or list of arrays
            pattern whose log likelihood is to be increased
        """

        delta_this = self._get_delta(out_pattern)
        if self.esn is not None:
            delta_esn = self.esn._get_delta(out_pattern)

        if delta_this is not None:
            self._update_parameters(delta_this)
        if self.esn is not None and delta_esn is not None:
            self.esn._update_parameters(delta_esn)

    def _get_mean(self):
        """Computing estimated mean

        Returns
        -------
        mu : array, shape (out_dim,)
            estimated mean
        """
        mu = self._get_conditional_negative_energy()

        return mu

    def _get_conditional_negative_energy(self):
        """Computing the fundamental output

        Returns
        -------
        array, shape (out_dim,)
            fundamental output
        """
        mu = VectorRegression._get_conditional_negative_energy(self)
        if "V" in self.variables:
            mu += amath.tensordot(self.e_trace, self.variables["V"], axes=2)

        if DEBUG:
            mu_naive = deepcopy(mu)
            for k in range(self.n_etrace):
                mu_naive = mu_naive \
                    + self.e_trace[k, :].dot(self.variables["V"][k])
            assert amath.allclose(mu, mu_naive), "ERROR: mu has a bug."

        # Echo state network
        if self.esn is not None:
            mu += self.esn._get_mean()

        return mu

    def _get_sample(self):
        """Returning mean as a sample, shape (out_dim,).
        LinearDyBM should be used as a deterministic model

        Returns
        -------
        array, shape (out_dim,)
            mu, estimated mean (deterministic)
        """
        return self._get_mean()

    def _time_reversal(self):
        """Making an approximately time-reversed LinearDyBM by transposing
        matrices.

        For discriminative learning, where in_dim != out_dim, time reversal
        would also implies that input and target is reversed
        """

        if self.esn is not None:
            # TODO: implement _time_reversal with ESN
            raise NotImplementedError("_time_reversal is not implemented with ESN")

        self._transpose_matrices()
        self._exchange_dimensions()

    def _transpose_matrices(self):
        """Making an approximately time-reversed LinearDyBM by transposing
        matrices.  Dimensions should be exchanged with _exchange_dimensions()
        """
        for i in xrange(self.n_etrace):
            self.variables["V"][i] = self.variables["V"][i].transpose()
            self.SGD.first["V"][i] = self.SGD.first["V"][i].transpose()
            self.SGD.second["V"][i] = self.SGD.second["V"][i].transpose()
        for i in xrange(self.len_fifo):
            self.variables["W"][i] = self.variables["W"][i].transpose()
            self.SGD.first["W"][i] = self.SGD.first["W"][i].transpose()
            self.SGD.second["W"][i] = self.SGD.second["W"][i].transpose()

    def _exchange_dimensions(self):
        """Exchanging in_dim and out_dim
        """
        out_dim = self.in_dim
        self.in_dim = self.out_dim
        self.out_dim = out_dim


class MultiTargetLinearDyBM(MultiTargetVectorRegression):

    """MultiTargetLinearDyBM is a building block for ComplexDyBM.
    MultiTargetLinearDyBM is similar to LinearDyBM but accepts
    multiple targets.  Namely, MultiTargetLinearDyBM can be trained to predict
    target patterns, :math:`(\mathbf{y}_1^{[t]}, \mathbf{y}_2^{[t]}, \ldots)`,
    at time :math:`t` from input time-series :math:`\mathbf{x}^{[:t-1]}` with

    .. math::

        \mathbf{b}_1
        + \sum_{\ell=1}^{L} \mathbf{W}_1^{[\ell]} \, \mathbf{x}^{[t-\ell]}
        + \sum_{k=1}^{K} \mathbf{V}_1^{[k]} \, \mathbf{e}^{[k]}

        \mathbf{b}_2
        + \sum_{\ell=1}^{L} \mathbf{W}_2^{[\ell]} \, \mathbf{x}^{[t-\ell]}
        + \sum_{k=1}^{K} \mathbf{V}_2^{[k]} \, \mathbf{e}^{[k]}

        \ldots

    .. todo:: Support the Echo State Network

    Parameters
    ----------
    in_dim : int
        Dimension of input time-series
    out_dims : list,
        List of the dimension of target time-series
    SGDs : list of the instances of SGD.SGD, optional
        List of the optimizer for the stochastic gradient method
    delay : int, optional
        length of the FIFO queue plus 1
    decay_rates : list, optional
        Decay rates of eligibility traces
    L1 : float, optional
        Strength of L1 regularization
    L2 : float, optional
        Strength of L2 regularization
    use_biases : list of boolean, optional
        Whether to use bias parameters
    sigma : float, optional
        Standard deviation of initial values of weight parameters
    insert_to_etrace : str, {"w_delay", "wo_delay"},  optional
        "w_delay" : Insert pattern observed d-1 time steps ago into
        eligibility traces.
        "wo_delay" : Insert the latest pattern into eligibility traces.
    random : arraymath.random
        random number generator
    """

    def __init__(self, in_dim, out_dims, SGDs=None, delay=2, decay_rates=[0.5],
                 L1=0, L2=0, use_biases=None, sigma=0,
                 insert_to_etrace="wo_delay", random=None):

        if SGDs is None:
            SGDs = [AdaGrad() for i in range(len(out_dims))]

        if len(out_dims) != len(SGDs):
            raise ValueError("out_dims and SGDs must have a common length")

        if use_biases is None:
            use_biases = [True] * len(out_dims)

        if random is None:
            random = amath.random.RandomState(0)

        self.layers = [LinearDyBM(in_dim, out_dim, delay, decay_rates, SGD,
                                  L1, L2, use_bias, sigma, insert_to_etrace,
                                  random=random)
                       for (out_dim, SGD, use_bias)
                       in zip(out_dims, SGDs, use_biases)]

        # Only layer 0 has internal states, which are shared among all layers
        for i in xrange(1, len(self.layers)):
            self.layers[i].fifo = self.layers[0].fifo
            self.layers[i].e_trace = self.layers[0].e_trace

        StochasticTimeSeriesModel.__init__(self)

    def _time_reversal(self):
        """
        Making an approximately time-reversed MultiTargetLinearDyBM by
        transposing matrices

        For discriminative learning, where in_dim != out_dim, time revierasal
        would also implies that input and target are exchanged
        """
        for layer in self.layers:
            layer._time_reversal()


class ComplexDyBM(StochasticTimeSeriesModel):

    """
    Complex DyBM with multiple visible layers
    Layers can have different delays, decay rates, and activations

    Parameters
    ----------
    delays : list of integers, length n_layers
        delays[l] corresponds to the delay of the l-th layer.
    decay_rates : list of lists of floats, length n_layers
        decay_rates[l] corresponds to the decay rates of eligibility traces
        of the l-th layer.
    activations : list of strings, 'linear' or 'sigmoid', length n_layers
        activations[l] corresponds to the activation unit of the l-th layer.
    in_dims : list of integers, length n_layers
        in_dims[l] corresponds to the dimension of input time-series
        of the l-th layer.
    out_dims : list of integers, length n_layers, optional
        out_dims[l] corresponds to the dimension of target time-series
        of the l-th layer.
        If None, set out_dims = in_dims.
    SGD : instance of SGD.SGD or list of instances of SGD.SGD, lengh n_layers if \
        list, optional.
        If list, SGD[l] corresponds to SGD for the l-th layer.
    L1 : float, optional
        Strength of L1 regularization.
    L2 : float, optional
        Strength of L2 regularization.
    use_biases : list of booleans, length n_layers
        Whether to use bias parameters in each layer.
    sigma : float, optional
        Standard deviation of initial values of parameters.
    insert_to_etrace : str, {"w_delay", "wo_delay"},  optional
        "w_delay" : Pattern observed d-1 time steps ago will be inserted
        into eligibility traces.
        "wo_delay" : The newest pattern will be inserted into
        eligibility traces.
    random : arraymath.random
        random number generator

    Attributes
    ----------
    activations : list of strings, 'linear' or 'sigmoid', length n_layers
        Activations
    out_dim : int
        The sum of all the out_dims
    n_layers : int
        The number of layers in complex DyBM.
    layers : list of LinearDyBM objects, length n_layers
        layers[l] corresponds to the l-th LinearDyBM.
    """

    def __init__(self, delays, decay_rates, activations, in_dims,
                 out_dims=None, SGD=None, L1=0., L2=0., use_biases=None,
                 sigma=0, insert_to_etrace="wo_delay", random=None):
        if insert_to_etrace not in ["w_delay", "wo_delay"]:
            raise ValueError("insert_to_etrace should be either 'w_delay' "
                             "or 'wo_delay'.")
        n_layers = len(delays)
        if out_dims is None:
            out_dims = in_dims
        if use_biases is None:
            use_biases = [[True] * n_layers] * n_layers
        if SGD is None:
            SGD = [AdaGrad() for _ in range(n_layers)]
        if SGD.__class__ is not list:
            SGD = [deepcopy(SGD) for _ in range(n_layers)]

        if not len(delays) == len(decay_rates) == len(activations) \
           == len(in_dims) == len(out_dims) == len(SGD) == len(use_biases):
            raise ValueError("delays, decay_rates, activations, in_dims, "
                             "out_dims, SGD, and use_biases must have the"
                             "same number of elements, each of which"
                             "corresponds to the parameters of each layer.")

        if random is None:
            random = amath.random.RandomState(0)

        self.activations = activations
        # out_dim: Total target dimension of this complex DyBM
        self.out_dim = sum(out_dims)
        self.layers = [MultiTargetLinearDyBM(dim, out_dims, deepcopy(SGD), delay,
                                             decay_rate, L1=L1, L2=L2,
                                             use_biases=use_bias, sigma=sigma,
                                             insert_to_etrace=insert_to_etrace,
                                             random=random)
                       for (delay, decay_rate, dim, sgd, use_bias)
                       in zip(delays, decay_rates, in_dims, SGD, use_biases)]
        self.n_layers = len(self.layers)
        self.insert_to_etrace = insert_to_etrace

        StochasticTimeSeriesModel.__init__(self)

    def init_state(self):
        """
        Initializing the state of each layer
        """
        for layer in self.layers:
            layer.init_state()

    def _update_state(self, in_patterns):
        """
        Updating the state of each layer

        Parameters
        ----------
        in_patterns : list of arrays, length n_layers
            in_patterns[l] : array, length in_dims[l].
                This corresponds to in_pattern for updating states of the \
                l-th layer.
        """
        for (layer, in_pattern) in zip(self.layers, in_patterns):
            layer._update_state(in_pattern)

    def _get_delta(self, out_patterns, expected=None, weightLLs=None):
        """
        Getting delta, how much we change parameters for each layer

        Parameters
        ----------
        out_patterns : list of arrays, length n_layers.
            out_patterns[l] : array, length out_dims[l]
                This corresponds to out_pattern of the l-th layer.
        expected : array, shape (out_dim,), optional
            Expected next output pattern
        weightLLs : list of boolean, length 2, optional
            weight[l] denotes whether to weigh the gradient with log likelihood

        Returns
        -------
        list of dicts, length n_layers.
            the l-th element corresponds to a dictionary of deltas with name \
            of a variable as a key.
        """
        if expected is None:
            # expected pattern with current parameters
            expected = self.predict_next()

        deltas = list()
        for layer in self.layers:
            d = layer._get_delta(out_patterns, expected, weightLLs)
            deltas.append(d)
        return deltas

    def _update_parameters(self, deltas):
        """
        Updating the parameters for each layer

        Parameters
        ----------
        deltas : list of dicts, length n_layers.
            deltas[l] corresponds to a dictionary of deltas of the l-th layer
            with name of a variable as a key.
        """
        for (layer, d) in zip(self.layers, deltas):
            layer._update_parameters(d)

    def get_LL(self, out_patterns):
        """
        Getting log likelihood of given out_patterns

        Parameters
        ----------
        out_patterns : list of arrays, length n_layers.
            out_patterns[l] : array, length out_dims[l]
                This corresponds to out_pattern of the l-th layer.

        Returns
        -------
        list
            The l-th element corresponds to the log likelihood
            of the l-th layer.
        """
        mu_list = self._get_conditional_negative_energy()

        LL = list()
        for layer, activation, mu, out_pattern in zip(self.layers,
                                                      self.activations,
                                                      mu_list,
                                                      out_patterns):
            if activation == "linear":
                # real valued prediction
                # same as prediction of Gaussian DyBM
                loglikelihood = None
            elif activation == "sigmoid":
                # Prediction in [0, 1]
                # For binary input, predicted value should be interpreted
                # as probability.
                # Same as the original binary DyBM
                loglikelihood = (- mu * out_pattern
                                 - amath.log(1. + amath.exp(-mu)))
                loglikelihood = amath.sum(loglikelihood)
            else:
                # TODO: implement other activations
                raise NotImplementedError("get_LL not implemented for " + activation)
            LL.append(loglikelihood)
        return LL

    def predict_next(self):
        """
        Predicting next pattern with the expected values

        Returns
        -------
        array, shape (out_dim,)
            prediction of out_pattern
        """
        return self._get_mean()

    def _get_mean(self):
        """
        Getting expected values

        Returns
        -------
        mean : list of arrays, length n_layers.
            mean[l] corresponds to the expected out_pattern for the l-th layer
        """

        mu_list = self._get_conditional_negative_energy()

        mean = list()
        for layer, activation, mu in zip(self.layers,
                                         self.activations,
                                         mu_list):
            if activation == "linear":
                # real valued prediction
                # same as prediction of Gaussian DyBM
                pred = mu
            elif activation in ["sigmoid", "sigmoid_deterministic"]:
                # prediction in [0, 1]
                # For binary input, predicted value should be interpreted
                # as probability.
                # Same as the original binary DyBM
                pred = sigmoid(mu)
            else:
                # TODO: implement other activations
                raise NotImplementedError("_get_mean not defined for activation:" + activation)
            mean.append(pred)

        return mean

    def _get_conditional_negative_energy(self):
        """
        Computing the conditional negative energy, summed up over all layers

        Returns
        -------
        list of array, length len(layers)
            list of conditional negative energy
        """
        mu_all = [amath.concatenate(layer._get_conditional_negative_energy())
                  for layer in self.layers]
        mu_all = amath.concatenate(mu_all).reshape((self.n_layers, self.out_dim))
        mu_sum = mu_all.sum(axis=0)

        start = 0
        mu_list = list()
        for layer in self.layers:
            in_dim = layer.get_input_dimension()
            mu = mu_sum[start:start + in_dim]
            mu_list.append(mu)
            start += in_dim

        return mu_list

    def _get_sample(self):
        """
        Sampling next values

        Returns
        -------
        list of arrays, length n_layers
            the l-th element corresponds to samples for the l-th layer.
        """
        means = self._get_mean()
        samples = list()
        for mean, activation in zip(means, self.activations):
            if activation in ["linear", "sigmoid_deterministic"]:
                # for linear activation, mean is always sampled
                sample = mean
            elif activation == "sigmoid":
                u = self.random.random_sample(mean.shape)
                sample = u < mean
            samples.append(sample)
        return samples

    def set_learning_rate(self, rates):
        """
        Setting the learning rates

        Parameters
        ----------
        rate : list of float, length n_layers
            learning rates
        """
        for layer in self.layers:
            layer.set_learning_rate(rates)


class BinaryDyBM(ComplexDyBM):

    """
    Binary (Bernoulli) DyBM, consisting of a visible Bernoulli layer

    Parameters
    ----------
    in_dim : integer
        The dimension of input time-series
    out_dim : integer, optional
        The dimension of target time-series
    delay : integer (>0), optional
        length of fifo queue plus one
    decay_rates : list of floats, optional
        Decay rates of eligibility traces of the l-th layer.
    SGD : object of SGD.SGD, optional
        Object of a stochastic gradient method
    L1 : float, optional
        Strength of L1 regularization
    L2 : float, optional
        Strength of L2 regularization
    sigma : float, optional
        Standard deviation of initial values of parameters
    insert_to_etrace : str, {"w_delay", "wo_delay"},  optional
        "w_delay" : Pattern observed d-1 time steps ago will be inserted
        into eligibility traces.
        "wo_delay" : The newest pattern will be inserted
        into eligibility traces.

    Attributes
    ----------
    n_etrace : int
        the number of eligibility traces
    len_fifo : int
        the length of FIFO queues (delay - 1)
    in_dim : int
        in_dim
    out_dim : int
        out_dim
    """

    def __init__(self, in_dim, out_dim=None, delay=2, decay_rates=[0.5],
                 SGD=None, L1=0, L2=0, sigma=0,
                 insert_to_etrace="wo_delay"):
        if not delay > 0:
            raise ValueError("delay must be a positive integer")
        if insert_to_etrace not in ["w_delay", "wo_delay"]:
            raise ValueError("insert_to_etrace should be either `w_delay` or "
                             "`wo_delay`.")

        if out_dim is None:
            out_dim = in_dim

        self.n_etrace = len(decay_rates)
        self.len_fifo = delay - 1
        self.in_dim = in_dim
        self.out_dim = out_dim
        if out_dim is None:
            out_dims = None
        else:
            out_dims = [out_dim]

        ComplexDyBM.__init__(self, [delay], [decay_rates], ["sigmoid"],
                             [in_dim], out_dims=out_dims, SGD=SGD, L1=L1, L2=L2,
                             use_biases=[[True]], sigma=sigma,
                             insert_to_etrace=insert_to_etrace)

    def get_input_dimension(self):
        """
        Getting the dimension of input sequence

        Returns
        -------
        in_dim : int
            dimension of input sequence
        """
        return self.in_dim

    def get_target_dimension(self):
        """
        Getting the dimension of target sequence

        Returns
        -------
        out_dim : int
            dimension of target sequence
        """
        return self.out_dim

    def set_learning_rate(self, rate):
        """
        Setting the learning rate

        Parameters
        ----------
        rate : float
            learning rate
        """
        ComplexDyBM.set_learning_rate(self, [rate])

    def _update_state(self, in_pattern):
        ComplexDyBM._update_state(self, [in_pattern])

    def _get_delta(self, out_pattern, expected=None):
        return ComplexDyBM._get_delta(self, [out_pattern])

    def get_predictions(self, in_seq):
        """
        Getting predicsions corresponding to given input sequence

        Parameters
        ----------
        in_seq : sequence or generator
            input sequence

        Returns
        -------
        list of arrays
           list of predictions
        """
        predictions = ComplexDyBM.get_predictions(self, in_seq)
        return [p[0] for p in predictions]


class GaussianBernoulliDyBM(ComplexDyBM):

    """
    Gaussian-Bernoulli DyBM, consisting of a visible Gaussian layer,
    and a hidden Bernoulli layer

    .. seealso:: Takayuki Osogami, Hiroshi Kajino, and Taro Sekiyama, \
    "Bidirectional learning for time-series models with hidden units", \
    ICML 2017.

    Parameters
    ----------
    delays : list of integers (>0), length 2
        delays[l] corresponds to the delay of the l-th layer.
    decay_rates : list of lists of floats, length 2
        decay_rates[l] corresponds to the decay rates of eligibility traces
        of the l-th layer.
    in_dims : list of integers, length 2
        in_dims[l] corresponds to the dimension of input time-series
        of the l-th layer.
    SGD : instance of SGD.SGD or list of instances of SGD.SGD, optional
        Instance of a stochastic gradient method
    L1 : float, optional
        Strength of L1 regularization
    L2 : float, optional
        Strength of L2 regularization
    sigma : float, optional
        Standard deviation of initial values of parameters
    insert_to_etrace : str, {"w_delay", "wo_delay"},  optional
        "w_delay" : Pattern observed d-1 time steps ago will be inserted
        into eligibility traces
        "wo_delay" : The newest pattern will be inserted
        into eligibility traces

    Attributes
    ----------
    out_dim : int
        the sum of all the out_dims
    n_etrace : int
        the number of eligibility traces
    len_fifo : int
        the length of FIFO queues (delay - 1)
    n_layers : int
        2, the number of layers
    layers : list of LinearDyBM objects, length 2
        layers[l] corresponds to the l-th LinearDyBM.
    visibility : list
        [True,False], visibility of layers
    """

    def __init__(self, delays, decay_rates, in_dims, SGD=None, L1=0.,
                 L2=0., sigma=0, insert_to_etrace="wo_delay"):
        if not len(delays) == len(decay_rates) == len(in_dims):
            raise ValueError("GaussianBernoulliDyBM only allows two layers")
        if not delays[0] == delays[1]:
            raise NotImplementedError("Current implementation only allows "
                                      "constant delay")
        if not len(decay_rates[0]) == len(decay_rates[1]):
            raise NotImplementedError("Current implementation only allows "
                                      "constant number of decay rates")
        if not delays[0] > 0:
            raise ValueError("delay must be a positive integer")
        if insert_to_etrace not in ["w_delay", "wo_delay"]:
            raise ValueError("insert_to_etrace should be either 'w_delay' or "
                             "'wo_delay'.")
        if SGD is None:
            SGD = [AdaGrad(), AdaGrad()]

        self.n_etrace = len(decay_rates[0])
        self.len_fifo = delays[0] - 1
        self.visibility = [True, False]
        # activations = ["linear", "sigmoid"]
        activations = ["linear", "sigmoid_deterministic"]
        use_bias = [[True, False], [False, False]]

        self.min_step_size = 1e-6  # to be updated adaptively
        self.max_step_size = 1.  # to be updated adaptively

        ComplexDyBM.__init__(self, delays, decay_rates, activations, in_dims,
                             in_dims, SGD, L1, L2, use_bias, sigma,
                             insert_to_etrace)

    def learn(self, in_generator, get_result=True):
        """
        Learning generator

        Parameters
        ----------
        in_generator : generator
            generator of input data for the visible Gaussian layer
        get_result : boolean, optional
            whether the accuracy of prediction during learning is yielded

        Returns
        ----------
        dict
            dictionary of
                "prediction": list of arraymath array, shape (out_dim, 1)
                "actual": list of arraymath array, shape (out_dim, 1)
                "error": list of float
                (squared error for each predicted pattern)
        """
        # [data, hidden samples]
        in_list_generator \
            = ListGenerator([in_generator, ElementGenerator(self, 1)])
        result = ComplexDyBM.learn(self, in_list_generator,
                                   get_result=get_result)
        if get_result:
            for key in result:
                result[key] = [x[0] for x in result[key]]
        return result

    def _get_delta(self, out_patterns, expected=None, weightLLs=None):
        """
        Getting delta, how much we change parameters for each layer

        Parameters
        ----------
        out_patterns : list of arrays, length 2.
            out_patterns[l] : array, length out_dims[l]
                This corresponds to out_pattern of the l-th layer.
        expected : list of arrays, length 2, optional
            expected[l] : array, length out_dims[l]
                out_pattern of the l-th layer expected by the current model
                to be computed if not given.
        weightLLs : list of boolean, length 2, optional.
            weight[l] denotes whether to weigh the gradient with log likelihood

        Returns
        -------
        list of dicts, length n_layers.
            The l-th element corresponds to a dictionary of deltas
            with name of a variable as a key.
        """
        return ComplexDyBM._get_delta(self, out_patterns, expected,
                                      [False, True])

    def get_predictions(self, in_generator):
        """
        Getting prediction of the visible layer, corresponding to a given input
        generator

        Parameters
        ----------
        in_generator : Generator
            input generator for the visible layer

        Returns
        -------
        list of arrays
            the i-th array is the i-th predicted pattern
        """
        in_generator = ListGenerator([in_generator, ElementGenerator(self, 1)])
        predictions_all_layers \
            = ComplexDyBM.get_predictions(self, in_generator)
        predictions = [p[0] for p in predictions_all_layers]
        return predictions

    def _time_reversal(self):
        """
        Making an approximately time-reversed GaussianBernoulliDyBM
        by transposing matrices
        """
        for v, length in zip(["V", "W"], [self.n_etrace, self.len_fifo]):
            for i in xrange(length):
                for j in xrange(2):
                    if v not in self.layers[j].layers[j].variables:
                        continue
                    # transposing visible to visible, hidden to hidden
                    self.layers[j].layers[j].variables[v][i] \
                        = self.layers[j].layers[j].variables[v][i].transpose()
                    self.layers[j].layers[j].SGD.first[v][i] \
                        = self.layers[j].layers[j].SGD.first[v][i].transpose()
                    self.layers[j].layers[j].SGD.second[v][i] \
                        = self.layers[j].layers[j].SGD.second[v][i].transpose()

                if v not in self.layers[1].layers[0].variables:
                    continue
                # transposing visible to hidden, hidden to visible
                tmp = deepcopy(
                    self.layers[1].layers[0].variables[v][i].transpose())
                self.layers[1].layers[0].variables[v][i] \
                    = self.layers[0].layers[1].variables[v][i].transpose()
                self.layers[0].layers[1].variables[v][i] = tmp

                tmp = deepcopy(
                    self.layers[1].layers[0].SGD.first[v][i].transpose())
                self.layers[1].layers[0].SGD.first[v][i] \
                    = self.layers[0].layers[1].SGD.first[v][i].transpose()
                self.layers[0].layers[1].SGD.first[v][i] = tmp

                tmp = deepcopy(
                    self.layers[1].layers[0].SGD.second[v][i].transpose())
                self.layers[1].layers[0].SGD.second[v][i] \
                    = self.layers[0].layers[1].SGD.second[v][i].transpose()
                self.layers[0].layers[1].SGD.second[v][i] = tmp

    def get_input_dimension(self):
        """
        Getting input dimension (size of the visible layer)

        Returns
        -------
        integer : input dimension
        """
        return self.layers[0].get_input_dimension()

    def get_target_dimension(self):
        """
        Getting target dimension (size of the visible layer)

        Returns
        -------
        integer : target dimension, which equals input dimension
        """
        return self.layers[0].get_input_dimension()

    def get_LL_sequence(self, in_seq, out_seq=None):
        # This is computationally hard
        raise NotImplementedError("get_LL_sequence not implemented for GaussianBernoulliDyBM")

    def _get_gradient_for_layer(self, out_pattern, target_layer):
        """
        Getting the gradient given out_pattern is given in target_layer

        Parameters
        ----------
        out_pattern : array
            target pattern
        target_layer : int
            index of target layer

        Returns
        -------
        list of array, length n_layers
            list of gradients for all layers
        """
        expected = self._get_mean()[target_layer]
        return [layer._get_gradient_for_layer(out_pattern, target_layer,
                                              expected=expected)
                for layer in self.layers]

    def _get_total_gradient(self, in_gen, out_gen=None):
        """
        Getting the gradient (sum of all stochastic gradients)

        Parameters
        ----------
        in_gen : generator
            input generator
        out_gen : generator, optional
            output generator

        Returns
        -------
        total_gradient : array
            gradient
        """
        self._reset_generators()

        total_gradient = None
        for in_pattern in in_gen:
            if out_gen is None:
                out_pattern = in_pattern
            else:
                out_pattern = out_gen.next()
            # get gradient
            gradient = self._get_gradient_for_layer(out_pattern, 0)
            # add the gradient into total_gradient
            if total_gradient is None:
                total_gradient = gradient
            else:
                for i in xrange(len(self.layers)):
                    for key in gradient[i]:
                        total_gradient[i][key] = total_gradient[i][key] + gradient[i][key]
            # update internal state
            sample = self._get_sample()[1]  # sample hidden activation
            self._update_state([in_pattern, sample])

        return total_gradient

    def _store_fifo(self):
        """
        Deepcopy fifo queues

        Returns
        -------
        original_fifo : list of fifo
           deepcopied fifo queues
        """
        original_fifo = [None, None]
        for i in range(2):
            original_fifo[i] = deepcopy(self.layers[i].layers[0].fifo)
        return original_fifo

    def _restore_fifo(self, original_fifo):
        """
        Set fifo queues

        Parameters
        ----------
        original_fifo : list of fifo
            new fifo queues
        """
        for i in range(2):
            self.layers[i].layers[0].fifo = original_fifo[i]

    def _apply_gradient(self, in_seq, out_seq=None, bestError=None):
        """
        Apply a gradient of a whole time-series with forward only

        Parameters
        ----------
        in_seq : list
            input sequence
        out_seq : list, optional
            target sequence
        bestError : float, optimal
            RMSE with the current model

        Returns
        -------
        bestError : float
            RMSE after applying the gradient
        """
        if out_seq is None:
            out_seq = in_seq

        in_gen = SequenceGenerator(in_seq)

        total_gradient = self._get_total_gradient(in_gen)

        best_variables = [deepcopy(layer.layers[0].variables) for layer in self.layers]

        if bestError is None:
            in_gen.reset()
            bestError = self._get_RMSE(in_gen, out_seq)

        # A simple line search for step size
        step_size = self.max_step_size
        while step_size > self.min_step_size:
            # tentatively apply gradient with a step size
            for i in xrange(len(self.layers)):
                for key in self.layers[i].layers[0].variables:
                    self.layers[i].layers[0].variables[key] \
                        = best_variables[i][key] + step_size * total_gradient[i][key]

            # evaluate error
            in_gen.reset()
            error = self._get_RMSE(in_gen, out_seq)

            # stop the line search if non-negligible improvement
            if error < bestError:
                best_variables = [deepcopy(layer.layers[0].variables)
                                  for layer in self.layers]
                bestError = error
                # increase the max step size if improves with the max step size
                if step_size == self.max_step_size:
                    self.max_step_size *= 2
                break

            step_size = step_size / 2.

        if step_size > self.min_step_size:
            return bestError
        else:
            for i in xrange(len(self.layers)):
                for key in self.layers[i].layers[0].variables:
                    self.layers[i].layers[0].variables[key] = best_variables[i][key]
            return None

    def _fit_by_GD(self, in_seq, out_seq=None, max_iteration=1000):
        """
        Fitting only with a forward learning

        Parameters
        ----------
        in_seq : list
            input sequence
        out_seq : list
            target sequence
        max_iteration : int, optional
            maximum number of iterations

        Returns
        -------
        bestError : float
            RMSE after learning
        """
        if out_seq is None:
            out_seq = in_seq

        in_gen = SequenceGenerator(in_seq)

        bestError = self._get_RMSE(in_gen, out_seq)
        for i in xrange(max_iteration):
            error = self._apply_gradient(in_seq, out_seq, bestError=bestError)
            if error is None:
                break
            else:
                bestError = error

        return bestError

    def _read(self, generator):
        """
        Read a squence from a generator, together with hidden activations,
        to update the state

        Parameters
        ----------
        generator : Generator
            Generator of a sequence
        """
        list_generator = ListGenerator([generator, ElementGenerator(self, 1)])
        for patterns in list_generator:
            self._update_state(patterns)

    def _get_bidirectional_gradient(self):
        """
        Getting the gradient for forward sequence and backward sequence

        Returns
        -------
        fwd_gradient : list
            gradient for forward sequence
        bwd_gradient : list
            gradient for backward sequence
        """
        self._reset_generators()

        self._read(self._fwd_warmup)
        fwd_gradient = self._get_total_gradient(self._fwd_train)

        self._time_reversal()
        self._read(self._bwd_warmup)
        bwd_gradient = self._get_total_gradient(self._bwd_train)
        self._time_reversal()

        return fwd_gradient, bwd_gradient

    def _get_RMSE(self, in_gen, target_seq):
        """
        Getting RMSE

        Parameters
        ----------
        in_gen : generator
            input generator
        target_seq : list
            target patterns

        Returns
        -------
        float
            RMSE
        """
        original_fifo = self._store_fifo()
        predictions = self.get_predictions(in_gen)
        self._restore_fifo(original_fifo)
        return RMSE(target_seq, predictions)

    def _reset_generators(self):
        """
        Resetting generators for training
        """
        self._fwd_warmup.reset()
        self._fwd_train.reset()
        self._bwd_warmup.reset()
        self._bwd_train.reset()

    def _get_bidirectional_error(self):
        """
        Getting weighted RMSE for forward sequence and backward sequence

        Returns
        -------
        error : float
            weighted RMSE
        """
        self._reset_generators()

        self._read(self._fwd_warmup)
        fwdError = self._get_RMSE(self._fwd_train, self._fwd_train_seq)

        if self._bwd_weight != 0:
            self._time_reversal()
            self._read(self._bwd_warmup)
            bwdError = self._get_RMSE(self._bwd_train, self._bwd_train_seq)
            self._time_reversal()
        else:
            bwdError = 0

        error = fwdError + self._bwd_weight * bwdError

        return error

    def _get_step_size_for_bidirectional(self, fwd_gradient, bwd_gradient):
        """
        Getting step size with line search

        Parameters
        ----------
        fwd_gradient: list
            graditn for forward sequence
        bwd_gradient : list
            gradient for backward sequence

        Returns
        -------
        step_size : float
           step size
        best_variables : dict
           variables after applying the gradient with the step size
        best_error : float
           RMSE after applying the gradient with the step size
        """
        # A simple line search for step size
        step_size = self.max_step_size
        while step_size > self.min_step_size:
            # tentatively apply gradient with a step size to the best variables
            self._restore_variables(self._best_variables)
            for i in xrange(len(self.layers)):
                for key in self.layers[i].layers[0].variables:
                    self.layers[i].layers[0].variables[key] \
                        = self.layers[i].layers[0].variables[key] \
                        + step_size * fwd_gradient[i][key]
            if self._bwd_weight != 0:
                self._time_reversal()
                for i in xrange(len(self.layers)):
                    for key in self.layers[i].layers[0].variables:
                        self.layers[i].layers[0].variables[key] \
                            = self.layers[i].layers[0].variables[key] \
                            + step_size * self._bwd_weight * bwd_gradient[i][key]
                self._time_reversal()

            # evaluate error
            error = self._get_bidirectional_error()

            # stop the line search if non-negligible improvement
            if error < self._best_error:
                self._best_variables = self._store_variables()
                self._best_error = error

                # update max_step_size
                if step_size == self.max_step_size:
                    # increase the max step size if improves with the max step size
                    self.max_step_size *= 10
                elif self.max_step_size > 10 * step_size:
                    # decrease max_step_size if much larger than improving step size
                    self.max_step_size = 10 * step_size
                break

            step_size = step_size / 2.

            # update min_step_size
            if step_size <= self.min_step_size and error > self._best_error * 1.001:
                self.min_step_size = self.min_step_size / 10
                print("reduced min_step_size to", self.min_step_size)
            if self.min_step_size < 1e-16:
                print("Too small a min_step_size", self.min_step_size)
                print("error", error)
                print("best_error", self._best_error)
                break

        if step_size < self.min_step_size:
            print("no improvement with minimum step size")
            error = self._get_bidirectional_error()

        return step_size, self._best_variables, self._best_error

    def _store_variables(self):
        """
        Deep copy variables

        Returns
        -------
        variables: list
            deepcopied variables
        """
        variables = [[None, None], [None, None]]
        for i, j in product(range(2), range(2)):
            variables[i][j] = deepcopy(self.layers[i].layers[j].variables)
        return variables

    def _restore_variables(self, variables):
        """
        Set variables

        Parameters
        ----------
        variables : list
            new variables
        """
        for i, j in product(range(2), range(2)):
            self.layers[i].layers[j].variables = deepcopy(variables[i][j])

    def _fit_bidirectional_by_GD(self, dataset, len_train, len_test, len_warm,
                                 bwd_weight, bwd_end, max_iteration=1000):
        """
        Fitting with bidirectional learning with gradient descent

        Parameters
        ----------
        dataset : list
            dataset
        len_train : int
            length of training dataset
        len_test : int
            length of test dataset
        len_warm : int
            length of warmup period
        bwd_weight : float
            weight of backward learning
        bwd_end : int
            when to stop bidirectional learning
        max_iteration : int
            when to stop learning

        Returns
        -------
        train_RMSE : float
            traning RMSE
        test_RMSE : float
            test RMSE
        """
        # Prepare dataset
        t0 = 0
        t1 = t0 + len_warm
        t2 = t1 + len_train
        t3 = t2 + len_warm
        t4 = t3 + len_test
        self._fwd_warmup = SequenceGenerator(dataset[t0:t1])
        self._fwd_train = SequenceGenerator(dataset[t1:t3])
        self._bwd_warmup = SequenceGenerator(dataset[t2:t3])
        self._bwd_train = SequenceGenerator(dataset[t0:t2])
        self._bwd_warmup.reverse()
        self._bwd_train.reverse()

        self._fwd_train_seq = self._fwd_train.to_list()
        self._bwd_train_seq = self._bwd_train.to_list()

        self._test_warmup = SequenceGenerator(dataset[t0:t3])
        self._test = SequenceGenerator(dataset[t3:t4])
        self._test_seq = self._test.to_list()

        self._bwd_weight = bwd_weight

        test_RMSE = list()
        train_RMSE = list()

        # Evaluate error before training
        baseline = baseline_RMSE(self._test_warmup.to_list()[-1], self._test_seq)

        self._test_warmup.reset()
        self._test.reset()
        self._read(self._test_warmup)
        predictions = self.get_predictions(self._test)
        rmse = RMSE(self._test_seq, predictions)
        test_RMSE.append(rmse)

        self._best_variables = self._store_variables()
        self._best_error = self._get_bidirectional_error()
        train_RMSE.append(self._best_error)

        for i in xrange(max_iteration):
            if i > max_iteration * bwd_end:
                self._bwd_weight = 0

            # get gradient
            fwd_gradient, bwd_gradient = self._get_bidirectional_gradient()

            for g in fwd_gradient:
                for key in g:
                    g[key] = g[key] / (len_train + len_warm)
            for g in bwd_gradient:
                for key in g:
                    g[key] = g[key] / (len_train + len_warm)

            # A simple line search for step size
            step_size, self._best_variables, self._best_error \
                = self._get_step_size_for_bidirectional(fwd_gradient, bwd_gradient)

            train_RMSE.append(self._best_error)

            self._test_warmup.reset()
            self._test.reset()
            self._read(self._test_warmup)
            predictions = self.get_predictions(self._test)
            rmse = RMSE(self._test_seq, predictions)
            test_RMSE.append(rmse)

            if step_size < self.min_step_size:
                break

        self._restore_variables(self._best_variables)

        return train_RMSE, test_RMSE
