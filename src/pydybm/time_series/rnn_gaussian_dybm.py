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

"""**Extending Gaussian DyBM with RNN model** - collectively refered
to as nonlinear DyBM model

The **RNNGaussianDyBM (RNN-G-DyBM)** model extends the **Gaussian
DyBM** model with a RNN hidden layer.

We consider a G-DyBM connected with a :math:`M`-dimensional RNN, whose
state vector changes dependent on a nonlinear feature mapping of its
own history and the :math:`N`-dimensional time-series input data
vector at time :math:`t-1`.

Where in, for most settings :math:`M > N`. Specifically, for
RNN-G-DyBM we consider the bias vector to be time-dependent:

.. math::

    \mathbf{b^{[t]}} = \mathbf{b^{[t-1]}} + \mathbf{A}^{\\top}\\Psi^{[t]}

Where, :math:`\\Psi^{[t]}` is the :math:`M \\times 1` dimensional
state vector at time :math:`t`, of a :math:`M`-dimensional RNN.
:math:`\mathbf{A}` is the :math:`M \\times N` dimensional learned
output weight matrix that connects the RNN state to the bias vector.
The RNN state is updated based on the input time-series vector
:math:`\mathbf{x}^{[t]}` as follows:

.. math::

    \\Psi^{[t]}
    = (1-\\rho)\\Psi^{[t-1]}
    + \\rho \mathcal{F}(\mathbf{W}_{rnn}\\Psi^{[t-1]}
    + \mathbf{W}_{in}\mathbf{x}^{[t]})

Here default is :math:`\mathcal{F}(x) = \mathtt{tanh}(x)`. This can be
replaced by ReLU or sigmoid nonlinearity.  :math:`0 < \\rho \leq1` is
a leak rate hyper-parameter of the RNN, which controls the amount of
memory in each unit of the RNN layer.  :math:`\mathbf{W}_{rnn}` and
:math:`\mathbf{W}_{in}` are the RNN weight matrix and projection of
the time series input to the RNN layer, respectively.  These are
initialized randomly and scaled to pre-specifed spectral radius.

Based on this formulation along with the eligibility traces and
weights in the DyBM model, predictions are made according to:

.. math::

    \mathbf{\\mu}^{[t]}
    = \mathbf{b}
    + \sum_{\\delta=1}^{d-1} \mathbf{W}^{[\\delta]} \, \mathbf{x}^{[t-\\delta]}
    + \sum_{k=1}^K \mathbf{U}_k \, \mathbf{\\alpha}_k^{[t-1]}

.. seealso:: Sakyasingha Dasgupta and Takayuki Osogami, "Nonlinear \
Dynamic Boltzmann Machines for Time-series Prediction," in \
Proceedings of the 31st AAAI Conference on Artificial Intelligence \
(AAAI-17), pages 1833-1839, San Francisco, CA, January 2017. \
http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14350/14343
"""

__author__ = "Sakyasingha Dasgupta <SDASGUP@jp.ibm.com>"


from ..time_series.vector_regression import VectorRegressionWithVariance, sigmoid
from ..base.sgd import AdaGrad
from .. import arraymath as amath
import six
import numpy as np


class GaussianDyBM(VectorRegressionWithVariance):

    """
    Gaussian DyBM

    For fixed variance, use ComplexDyBM with a single layer for Gaussian DyBM
    with fixed unit variance

    Parameters
    ----------
    in_dim : int
        dimension of input time-series
    out_dim : int, optional
        dimension of target time-series
    delay : int, optional
        length of fifo queue plus one
    decay_rates : list, optional
        decay rates of eligibility traces
    SGD : object of SGD.SGD, optional
        object of a stochastic gradient method
    L1 : float, optional
        strength of L1 regularization
    L2 : float, optional
        strength of L2 regularization
    insert_to_etrace : str, {"w_delay", "wo_delay"},  optional
        "w_delay" : insert pattern observed d-1 time steps ago into
        eligibility traces
        "wo_delay" : insert the latest pattern into eligibility traces

    Attributes
    ----------
    in_dim : int
        in_dim
    out_dim : int
        out_dim
    n_etrace : int
        the number of eligibility traces
    len_fifo : int
        the length of FIFO queues (delay - 1)
    decay_rates : array, shape (n_etrace, 1)
        decay rates of eligibility traces
    variables : dict
        dictionary of model parameters
    variables["W"] : array, shape (len_fifo, in_dim, out_dim)
        variables["W"][l] corresponds to the weight from the input observed at
        time step t - l - 1 to the mean at time step t (current time).
    variables["b"] : array, shape (out_dim,)
        variables["b"] corresponds to the bias to out_pattern.
    variables["s"] : array, shape (out_dim,)
        variables["s"][n] corresponds to the standard deviation of
        out_pattern[n] (or scale parameter in other words).
    variables["V"] : array, shape (n_etrace, in_dim, out_dim)
        variables["V"][k] corresponds to the weight from the k-th eligibility
        trace to the mean.
    fifo : deque
        FIFO queue storing len_fifo in_patterns, each in_pattern has shape (in_dim,).
    e_trace : array, shape (n_etrace, in_dim)
        e_trace[k, :] corresponds to the k-th eligibility trace.
    """

    def __init__(self, in_dim, out_dim=None, delay=2, decay_rates=[],
                 SGD=None, L1=0., L2=0., insert_to_etrace="wo_delay"):
        if insert_to_etrace not in ["w_delay", "wo_delay"]:
            raise ValueError("insert_to_etrace should be either `w_delay` "
                             "or `wo_delay`.")
        self.n_etrace = len(decay_rates)
        self.decay_rates = amath.array(decay_rates).reshape((self.n_etrace, 1))
        order = delay - 1
        VectorRegressionWithVariance.__init__(self, in_dim, out_dim, order,
                                              SGD, L1, L2)
        self.variables["V"] = amath.zeros(
            (self.n_etrace, self.in_dim, self.out_dim), dtype=float)

        if SGD is None:
            SGD = AdaGrad()
        self.SGD = SGD.set_shape(self.variables)  # resetting SGD
        self.L1["V"] = L1
        self.L2["V"] = L2
        self.insert_to_etrace = insert_to_etrace

    def init_state(self):
        """
        initializing FIFO queue and eligibility traces
        """
        VectorRegressionWithVariance.init_state(self)
        # eligibility trace
        self.e_trace = amath.zeros((self.n_etrace, self.in_dim))

    def _update_state(self, in_pattern):
        """
        updating FIFO queues and eligibility traces by appending in_pattern

        Parameters
        ----------
        in_pattern : array, shape (in_dim,)
            in_pattern for updating states
        """
        assert in_pattern.shape == (self.in_dim,), "in_pattern must have shape (in_dim,)"

        popped_in_pattern \
            = VectorRegressionWithVariance._update_state(self, in_pattern)

        if self.insert_to_etrace == "wo_delay":
            self.e_trace = self.e_trace * self.decay_rates + in_pattern
        elif self.insert_to_etrace == "w_delay":
            self.e_trace = self.e_trace * self.decay_rates + popped_in_pattern
        else:
            print("not implemented.")
            exit(-1)

    def _get_gradient(self, out_pattern, expected=None):
        """
        computing the gradient of log likelihood

        Parameters
        ----------
        out_pattern : array, length out_dim
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

        gradient = VectorRegressionWithVariance._get_gradient(self, out_pattern)

        if self.n_etrace > 0:
            # TODO: dx in the following has already been computed in
            # VectorRegressionWithVariance._get_gradient
            if expected is None:
                expected = self._get_mean()
            NATURAL_GRADIENT = True
            if NATURAL_GRADIENT:
                dx = out_pattern - expected  # natural gradient
            else:
                dx = (out_pattern - expected) / self.variables["s"]**2
            gradient["V"] = amath.op.mult_2d_1d_to_3d(self.e_trace, dx)

        self.SGD.apply_L2_regularization(gradient, self.variables, self.L2)

        return gradient

    def _get_mean(self):
        """
        computing estimated mean

        Returns
        -------
        mu : array, shape (out_dim,)
            estimated mean
        """
        mu = VectorRegressionWithVariance._get_mean(self)
        for k in range(self.n_etrace):
            # TODO einsum
            mu = mu + self.e_trace[k, :].dot(self.variables["V"][k])
        return mu


class RNNGaussianDyBM(GaussianDyBM):

    """
    Generative and Discriminative Learning using
    Gaussian DyBM with nonlinear RNN layer for updating bias

    Parameters
    ----------
    in_dim : int
        dimension of input time-series
    rnn_dim : int
        dimension of RNN layer
    out_dim : int, optional
        dimension of target time-series
    delay : int, optional
        length of fifo queue plus one
    leak : float
        RNN layer leak rate
    sparsity: float
        connection probability of RNN layer weight matrix
    spectral_radius: float
        scaling parameter of RNN layer weight matrix
    activation : string
       activation function type for RNN layer units
       {tanh, relu, sigmoid, linear}
    decay_rates : list, optional
        decay rates of eligibility traces
    SGD : object of SGD.SGD, optional
        object of a stochastic gradient method
    L1 : float, optional
        strength of L1 regularization
    L2 : float, optional
        strength of L2 regularization

    Attributes
    ----------
    in_dim : int
        in_dim
    rnn_dim : int
        rnn_dim
    out_dim : int
        out_dim
    n_etrace : int
        the number of eligibility traces
    si : array, shape (rnn_dim,)
        si stores the per time step state of the RNN layer
    decay_rates : array, shape (n_etrace,1)
        decay rates of eligibility traces
    variables : dict
        dictionary of model parameters
    variables["W"] : array, shape (len_fifo, in_dim, out_dim)
        variables["W"][l] corresponds to the weight from the input observed at
        time step t - l - 1 to the mean at time step t (current time).
    variables["b"] : array, shape (out_dim,)
        variables["b"] corresponds to the bias to out_pattern.
    variables["s"] : array, shape (out_dim,)
        variables["s"][n] corresponds to the standard deviation of
        out_pattern[n] (or scale parameter in other words).
    variables["V"] : array, shape (n_etrace, in_dim, out_dim)
        variables["V"][k] corresponds to the weight from the k-th eligibility
        trace to the mean.
    variables["A"] : array, shape (out_dim,rnn_dim)
        variables["A"] corresponds to the output weights from the RNN layer
    fifo : deque
        FIFO queue storing len_fifo in_patterns, each in_pattern has shape (in_dim,).
    e_trace : array, shape (n_etrace, in_dim)
        e_trace[k, :] corresponds to the k-th eligibility trace.
    """

    def __init__(self, in_dim, out_dim, rnn_dim, spectral_radius,
                 sparsity, delay=2, decay_rates=[0.5], leak=1.0, SGD=None,
                 random_seed=None, L1=0.0, L2=0.0):

        # if out_dim is None:
        # out_dim = in_dim

        if random_seed is not None:
            amath.random.seed(random_seed)

        self.rnn_dim = rnn_dim
        self.delay = delay
        self.decay_rates = decay_rates
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius

        self.leak = leak         # RNN leak term for each unit
        self.si = amath.zeros((self.rnn_dim,))  # RNN state vector

        if SGD is None:
            SGD = AdaGrad()

        GaussianDyBM.__init__(self, in_dim, out_dim, delay, decay_rates, SGD,
                              L1=L1, L2=L2)

        # Default setting of RNN-output matrix to zero
        self.variables["A"] = amath.zeros((self.out_dim, self.rnn_dim),
                                          dtype=float)

        # Use this setting in a task specific manner
        # self.variables["A"] = np.random.normal(0, 0.1,(self.out_dim, self.rnn_dim)) \
        #    // np.sqrt(self.rnn_dim)
        # self.variables["A"] = np.random.uniform(0, 0.1,(self.out_dim, self.rnn_dim)) \
        #    // np.sqrt(self.rnn_dim)

        self.SGD = SGD.set_shape(self.variables)  # resetting SGD

        self.L1["A"] = 1.0    # change L1 strength in a task specific manner
        self.L2["A"] = 0.0

        # initialize Recurrent layer weight matrix at edge of chaos
        self.Wrec = amath.random.normal(0, 1, (self.rnn_dim, self.rnn_dim)) \
                    / np.sqrt(self.rnn_dim)
        # delete the fraction of connections given by (self.sparsity):
        amath.assign_if_true(
            self.Wrec,
            amath.random.rand(*self.Wrec.shape) < self.sparsity,
            0
        )
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eig(amath.to_numpy(self.Wrec))[0]))
        # rescale them to reach the requested spectral radius:
        self.Wrec = self.Wrec * (self.spectral_radius / radius)
        # initialise input to RNN weight matrix at edge of chaos
        self.Win = amath.random.normal(0, 0.1, (self.rnn_dim, self.in_dim)) \
                   / amath.sqrt(self.in_dim)

        self.init_state()

    def init_state(self):
        """
        initializing FIFO queue and eligibility traces
        """
        GaussianDyBM.init_state(self)

    def _update_state(self, in_pattern):
        """
        Updating FIFO queues, RNN state and eligibility traces by appending
        in_pattern

        Parameters
        ----------
        in_pattern : array, shape (in_dim,)
            in_pattern for updating states
        """
        assert in_pattern.shape == (self.in_dim,), "in_pattern must have shape (in_dim,)"

        # update fifo and eligibility traces
        GaussianDyBM._update_state(self, in_pattern)

        # update the RNN state
        Win_in = amath.dot(self.Win, in_pattern.reshape(-1, 1)).reshape(-1,)
        Wrec_si = amath.dot(self.Wrec, self.si.reshape(-1, 1)).reshape(-1,)
        self.si *= 1 - self.leak
        self.si += self.leak * amath.tanh(Win_in + Wrec_si)

        # update the DyBM bias
        self.variables["b"] += amath.dot(self.variables["A"],
                                         self.si.reshape(-1, 1)).reshape(-1,)

    def ReLU(self, state):
        """
        Rectified linear nonlinearity for RNN units

        Parameters
        ----------
        state : array, shape (rnn_dim,)
            RNN state vector
        """
        value = amath.log(1 + amath.exp(state))
        return value

    def _sigmoid(self, state):
        """
        Sigmoidal nonlinearity for RNN units

        Parameters
        ----------
        state : array, shape (rnn_dim,)
            RNN state vector
        """
        return sigmoid(state)

    def _get_gradient(self, out_pattern, expected=None):
        """
        computing the gradient of log likelihood

        Parameters
        ----------
        out_pattern : array, length out_dim
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

        gradient = GaussianDyBM._get_gradient(self, out_pattern)

        if expected is None:
            expected = self._get_mean()

        dx = (out_pattern - expected) / self.variables["s"]**2
        # dx = dx.reshape((self.in_dim,))
        dx = dx.ravel()  # reshape((self.out_dim,))

        temp_si = self.si.reshape((self.rnn_dim, 1))
        # temp_dx = dx.reshape((self.in_dim, 1))
        temp_dx = dx.reshape((self.out_dim, 1))

        gradient["A"] = amath.dot(temp_dx, amath.transpose(temp_si))

        self.SGD.apply_L2_regularization(gradient, self.variables, self.L2)

        return gradient

    def _get_mean(self):
        """
        computing estimated mean

        Returns
        -------
        mu : array, shape (out_dim,)
            estimated mean
        """
        mu = GaussianDyBM._get_mean(self)

        return mu

    def compute_RMSE(self, pattern):
        """
        Compute the root mean squared error between current input pattern and
        DyBM prediction

        Parameters
        ----------
        pattern : array, shape (out_dim,)

        """
        if not pattern.shape == (self.out_dim,):
            raise ValueError("pattern must have shape (out_dim,)")

        expected = GaussianDyBM.predict_next(self)

        return amath.root_mean_square_err(expected, pattern)

    def set_learning_rate(self, rate):
        """
        Set learning rate of SGD.

        Parameters
        ----------
        rate : float
            Learning rate.
        """

        self.SGD.set_learning_rate(rate)
        pass

