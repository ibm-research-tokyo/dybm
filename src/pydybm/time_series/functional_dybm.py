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

"""Implementation of a functional DyBM.

A functional DyBM (F-DyBM) models the dynamics of uncountably many
neurons distributed in the :math:`D`-dimensional space,
:math:`\mathcal{X}\subseteq{\mathbb{R}^{D}}`, which we call a *feature
space*.  Each point :math:`x\in\mathcal{X}` has a neuron whose output
at time :math:`t` is :math:`f^{[t]}(x)`.  F-DyBM models
:math:`f^{[t]}(x)` given :math:`f^{[-\infty,t-1]}(x)` as

.. math::

    f^{[t]} \mid f^{[-\infty,t-1]} \sim
    \mathcal{GP}(\mu^{[t]}(x), K(x,{x}^\prime)),

where :math:`K\colon{\mathcal{X}} {\\times} {\mathcal{X}}
{\\rightarrow} {\mathbb{R}}` is a kernel function, and

.. math::

    \mu^{[t]}(x)
    = b(x)
    + \sum_{\delta=1}^{d-1} \int w^{[\delta]}(x, x^{\prime})
    f^{[t-\delta]}(x^\prime) dx^{\prime}
    + \sum_{k=1}^K \int
    u_k(x, x^{\prime}) \\alpha_k^{[t-1]}(x^{\prime}) dx^{\prime},

    \\alpha_k^{[t-1]}(x^\prime)
    = \sum_{\delta=d}^{\infty} \lambda_k^{\delta - d} f^{[t-\delta]}(x^\prime).

in the case of ``insert_to_delay="w_delay"``, and

.. math::

    \\alpha_k^{[t-1]}(x^\prime)
    = \sum_{\delta=1}^{\infty} \lambda_k^{\delta - 1} f^{[t-\delta]}(x^\prime).

in the case of ``insert_to_delay="wo_delay"``.

* Input at each time step
    :math:`(X^{[t]}, f^{[t]}(X^{[t]}))`

    * :math:`X^{[t]} = [x_1^{[t]},\dots,x_N^{[t]}]^{\\top}` is a set of observation points, ``loc``.

    * :math:`f^{[t]}(X^{[t]}) = [f^{[t]}(x_1^{[t]}), \dots, f^{[t]}(x_N^{[t]})]^{\\top}` is a set of observations at time :math:`t`, ``pattern``.



* ``FunctionalDyBM`` class implements the method presented in the IJCAI-17 paper.

* ``FuncLinearDyBM`` class implements a heuristic method to train LinearDyBM using the data in the above format, which is used for comparison (G-DyBM).

.. seealso:: Hiroshi Kajino: A Functional Dynamic Boltzmann Machine, \
Proceedings of the International Joint Conference on Artificial Intelligence 2017 (IJCAI-2017).
"""

__author__ = "Hiroshi Kajino"
__version__ = "1.1"
__date__ = "20th July 2016"

import argparse
from .. import arraymath as amath
from ..time_series.dybm import LinearDyBM
from ..base.sgd import RMSProp
from functools import partial
from collections import deque
from six.moves import xrange
from copy import deepcopy

DEBUG = False


def _check_pattern_loc_consistency(pattern, loc, dim):
    """ Check pattern and loc are consistent.

    Parameters
    ----------
    pattern: array, shape (n_obs,)
        Pattern on which we compute RMSE using the prediction for it.
    loc: array, shape (n_obs, dim)
        Locations of each observation of the pattern,
        i.e., pattern[i] is observed at loc[i].
    """
    pattern = pattern.ravel()
    _check_loc_consistency(loc, dim)
    if len(pattern) != loc.shape[0]:
        raise ValueError("pattern and loc have inconsistent numbers of observations.")


def _check_loc_consistency(loc, dim):
    if loc.ndim != 2:
        raise ValueError("loc must be 2-dimensional array.")
    if loc.shape[1] != dim:
        raise ValueError("loc has inconsistent dimension.")


class FunctionalDyBM(object):
    """A class of a functional DyBM.
    Weight functions :math:`w^{[\delta]}(x,x^\prime)` and :math:`u_l(x,x^\prime)` is approximated by

    .. math::

        w^{[\delta]}(x,x^\prime) = K(x, P_1) W^{[\delta]} K(P_2, x^\prime),

        u_l(x,x^\prime) = K(x, P_1) U_l K(P_2, x^\prime),

    where :math:`W^{[d]}` and :math:`U_l` correspond to ``variables["W"][d]`` and ``variables["U"][l]``, and
    :math:`P_1` and :math:`P_2` correspond to ``anc_points[0]`` and ``anc_points[1]`` respectively.

    Parameters
    ----------
    dim: int
        Dimension of the feature space.
    anc_points: array, shape (n_anc, dim), or tuple of arrays, \
    shapes (n_anc_basis, dim) and (n_anc_data, dim)
        Anchor points, on which we implement a finite-dimensional DyBM
        (with ``n_anc`` units).
    delay: int
        Conduction delay. previous ``delay - 1`` patterns are used as regressors.
    decay_rates: list, length n_elg
        Decay rates of eligibility traces, each of which is a float in [0, 1).
        ``n_elg`` stands for "the number of eligibility traces".
    noise_var: float, optional
        A variance parameter of a Gaussian distribution that models a noise on
        observation.
    lmbd_reg: dict, optional
        A dictionary storing regularization coefficients for model parameters.
        ``lmbd_reg.keys() == ["b", "W", "V"]``.
    ker_paras: dict, optional
        Parameters of the kernel.
        For example, ``ker_paras = {"ker_type": "rbf", "gamma": 1.0}``.
        For details, see `here <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics>`_.
        ``"ker_type"`` is either ``'polynomial'``, ``'laplacian'``, ``'linear'``,
        ``'additive_chi2'``, ``'sigmoid'``, ``'chi2'``, ``'rbf'``, ``'cosine'``, ``'poly'``.
    insert_to_etrace: str, {"w_delay", "wo_delay"},  optional
        ``"w_delay"``: insert pattern observed d-1 time steps ago into eligibility
        traces
        ``"wo_delay"``: the newest pattern will be inserted into eligibility traces
    SGD: object of SGD.SGD, optional
        Object of a stochastic gradient method.
        If ``None``, we update parameters, fixing the learning rate.
    learning_rate: float, optional
        Learning rate for SGD.
    learn_anc_basis: bool, optional
        if ``True``, self.anc_basis will be learned from data online.

    Attributes
    ----------
    anc_basis : array, shape (n_anc_basis, dim)
        ``anc[i_anc, :]`` corresponds to the ``i_anc``-th anchor point. fixed.
        Initialized by ``anc_points[0]``.
    anc_data : array, shape (n_anc_data, dim)
        ``anc[i_anc, :]`` corresponds to the ``i_anc``-th anchor point. fixed.
        Initialized by ``anc_points[1]``.
    variables : dict
        Dictionary of model parameters.
    variables["W"] : array, shape (delay-1, n_anc_basis, n_anc_data)
        ``variables["W"][d]`` corresponds to the weights from pattern at time
        step ``t - d`` to the mean at time step ``t`` (current time).
    variables["b"] : array, shape (n_anc_basis,)
        ``variables["b"]`` corresponds to the bias.
    variables["V"] : array, shape (n_elg, n_anc_basis, n_anc_data)
        ``variables["V"][k]`` corresponds to the weights from the ``k``-th eligibility
        trace to the mean.
    mu : array, shape (n_anc_basis,)
        Values of mu.
    fifo : array, shape (delay-1, n_anc_data)
        FIFO queue storing ``delay - 1`` patterns, where each pattern has shape
        ``(n_anc_data,)``.
        ``fifo[0]`` is the newest, and ``fifo[-1]`` is the oldest.
    e_trace : array, shape (n_elg, n_anc_data)
        ``e_trace[k, :]`` corresponds to the ``k``-th eligibility trace.
    """

    def __init__(self, dim, anc_points, delay, decay_rates,
                 noise_var=0.1, lmbd_reg={"b": 0.0, "W": 0.0, "V": 0.0},
                 ker_paras=None, insert_to_etrace="w_delay",
                 SGD=None, learning_rate=None,
                 learn_anc_basis=False):
        if type(anc_points) == tuple:
            self.anc_basis = anc_points[0]
            self.anc_data = anc_points[1]
        else:
            self.anc_basis = anc_points
            self.anc_data = anc_points
        if not dim == self.anc_basis.shape[1]:
            raise ValueError("inconsistent dimension: "
                             "{}, {}".format(dim, self.anc_basis.shape[1]))
        if not dim == self.anc_data.shape[1]:
            raise ValueError("inconsistent dimension: "
                             "{}, {}".format(dim, self.anc_data.shape[1]))
        if insert_to_etrace not in ["w_delay", "wo_delay"]:
            raise ValueError("insert_to_etrace should be either 'w_delay' "
                             "or 'wo_delay'.")
        self.insert_to_etrace = insert_to_etrace
        self.dim = dim
        self.n_anc_basis = self.anc_basis.shape[0]
        self.n_anc_data = self.anc_data.shape[0]
        self.delay = delay
        self.decay_rates = decay_rates
        self.n_elg = len(decay_rates)

        if self.n_elg >= 1:
            self.variables = {"W": amath.zeros((delay-1, self.n_anc_basis,
                                                self.n_anc_data)),
                              "b": amath.zeros(self.n_anc_basis),
                              "V": amath.zeros((self.n_elg, self.n_anc_basis,
                                                self.n_anc_data))}
            # L1_coef: Not used in the current implementation.
            self.L1_coef = {"W": 0.0, "b": 0.0, "V": 0.0}
        else:
            self.variables = {"W": amath.zeros((delay-1, self.n_anc_basis,
                                                self.n_anc_data)),
                              "b": amath.zeros(self.n_anc_basis)}
            # L1_coef: Not used in the current implementation.
            self.L1_coef = {"W": 0.0, "b": 0.0}
        self.noise_var = noise_var
        self.lmbd_reg = lmbd_reg

        # set kernel function
        if ker_paras is None:  # rbf kernel
            ker_type = "rbf"
            self.ker_paras = {"gamma": 1.0}
            self.ker = partial(amath.kernel_metrics["rbf"], **self.ker_paras)
        else:
            self.ker_paras = ker_paras
            ker_paras = deepcopy(self.ker_paras)
            ker_type = ker_paras.pop("ker_type")
            self.ker = partial(amath.kernel_metrics[ker_type],
                               **ker_paras)
        # to check ker_paras is consistent or not.
        _ = self.ker(amath.zeros((1, 1)), amath.zeros((1, 1)))

        if learn_anc_basis:
            if not ker_type == "rbf":
                raise NotImplementedError("only rbf kernel is supported for "
                                          "`learn_anc_basis` option.")
            self.variables["anc_basis"] = self.anc_basis
            self.L1_coef["anc_basis"] = 0.0
        self.learn_anc_basis = learn_anc_basis
        if SGD is None:
            self.SGD = RMSProp()
            self.SGD.set_shape(self.variables)
            if learning_rate is None:
                self.learning_rate = 10 ** (-5)
            else:
                self.learning_rate = learning_rate
        else:
            self.SGD = SGD
            self.SGD.set_shape(self.variables)
            if learning_rate is not None:
                self.SGD.set_learning_rate(learning_rate)

        self.init_state()

    def learn_one_step(self, pattern, loc):
        """ Update model parameters using a new observation ``(pattern, loc)``.
        After executing ``learn_one_step()``, it is ready to predict the next pattern using
        ``predict_next()``.

        Parameters
        ----------
        pattern: array, shape (n_obs,)
            Pattern used to update the state.
        loc: array, shape (n_obs, dim)
            Locations of each observation,
            i.e., ``pattern[i]`` is observed at ``loc[i]``.
        """
        _check_pattern_loc_consistency(pattern, loc, self.dim)
        delta = self._get_delta(pattern, loc)
        self._update_parameters(delta)
        self._update_state(pattern, loc)

    def learn(self, pattern_seq, loc_seq, init_state=False):
        """ This method trains the model by one-pass SGD on the given data.
        After executing this method, the model is ready for predicting the next pattern.

        Parameters
        ----------
        pattern_seq : list
            list of patterns.
        loc_seq : list
            list of locations.
        init_state : bool
            if True, FIFO and eligibility traces are initialized before learning.
        """
        if len(pattern_seq) != len(loc_seq):
            raise ValueError("lengths of pattern_seq and loc_seq are inconsistent.")
        if init_state:
            self.init_state()
        for each_pattern, each_loc in zip(pattern_seq, loc_seq):
            self.learn_one_step(each_pattern, each_loc)

    def predict_next(self, loc):
        """
        Predict the next pattern in a deterministic manner.

        Parameters
        ----------
        loc: array, shape (n_loc, dim)
            Locations on which we want to predict the pattern.

        Returns
        -------
        array, shape (n_loc,)
            Predictions.
        """
        _check_loc_consistency(loc, self.dim)
        if self.learn_anc_basis:
            self.anc_basis = self.variables["anc_basis"]
        return self.ker(loc, self.anc_basis).dot(
            self.mu.reshape(-1, 1)).reshape(-1,)

    def compute_RMSE(self, pattern, loc):
        """
        Compute the RMSE between the prediction and the actual observation.
        Given the model prediction :math:`\mu^{[t+1]}(x)` and the actual
        observation :math:`(X^{[t+1]},f^{[t+1]}(X^{[t+1]}))`, this method
        outputs
        :math:`\dfrac{1}{\sqrt{\mathrm{n\_obs}}}\|\mu^{[t+1]}(X^{[t+1]})
        - f^{[t+1]}(X^{[t+1]})\|_{2}`.

        Parameters
        ----------
        pattern: array, shape (n_obs,)
            Pattern on which we compute RMSE using the prediction for it.
        loc: array, shape (n_obs, dim)
            Locations of each observation of the pattern,
            i.e., ``pattern[i]`` is observed at ``loc[i]``.

        Returns
        -------
        float
            RMSE between pattern and expected pattern.
        """
        _check_pattern_loc_consistency(pattern, loc, self.dim)
        pattern = pattern.ravel()
        n_obs = len(pattern)
        expected_pat = self.predict_next(loc)
        return amath.root_mean_square_err(expected_pat, pattern)

    def get_LL(self, pattern, loc):
        """
        Get the conditional log likelihood of a pattern in the next time step.
        The pattern in the argument has not been used for training.

        Parameters
        ----------
        pattern: array, shape (n_obs,)
            Pattern on which we compute the log likelihood using the prediction
            for it.
        loc: array, shape (n_obs, dim)
            Locations of each observation of the pattern,
            i.e., ``pattern[i]`` is observed at ``loc[i]``.

        Returns
        -------
        float
            log pdf (probability density function) of the pattern
        """
        _check_pattern_loc_consistency(pattern, loc, self.dim)
        pattern = pattern.ravel()
        n_obs = len(pattern)

        mu_loc = self.predict_next(loc)
        cov_mat = self.ker(loc, loc) + self.noise_var * amath.identity(n_obs)
        return amath.stats_multivariate_normal_logpdf(pattern,
                                                      mean=mu_loc,
                                                      cov=cov_mat)

    def init_state(self):
        """
        Initialize FIFO, eligibility traces, and initial prediction
        """
        self.fifo = amath.zeros((self.delay-1, self.n_anc_data))
        if self.n_elg >= 1:
            self.e_trace = amath.zeros((self.n_elg, self.n_anc_data))

            self.mu \
                = self.variables["b"] \
                + amath.tensordot(self.fifo, self.variables["W"], axes=2) \
                + amath.tensordot(self.e_trace, self.variables["V"], axes=2)

        else:
            self.mu = self.variables["b"] \
                + amath.tensordot(self.fifo, self.variables["W"], axes=2)

    def _update_state(self, pattern, loc):
        """
        Update FIFO and eligibility traces.

        Parameters
        ----------
        pattern: array, shape (n_obs,)
            Pattern used to update the state.
        loc: array, shape (n_obs, dim)
            Locations of each observation,
            i.e., ``pattern[i]`` is observed at ``loc[i]``.
        """
        _check_pattern_loc_consistency(pattern, loc, self.dim)
        pattern = pattern.ravel()
        if self.learn_anc_basis:
            self.anc_basis = self.variables["anc_basis"]

        n_obs = len(pattern)

        ker_anc_data_loc = self.ker(self.anc_data, loc)
        ker_anc_basis_loc = self.ker(self.anc_basis, loc)
        ker_loc_loc = self.ker(loc, loc)
        ker = self.ker(self.anc_data, self.anc_basis)
        solve = amath.linalg_solve(
            ker_loc_loc
            + self.noise_var
            * amath.identity(n_obs),
            pattern
            - self.mu.dot(ker_anc_basis_loc))
        f_t = ker.dot(self.mu.reshape(-1, 1)).reshape(-1,) \
              + ker_anc_data_loc.dot(solve.reshape(-1, 1)).reshape(-1,)

        if self.n_elg >= 1:
            # update fifo and eligibility traces
            if self.insert_to_etrace == "w_delay":
                self.e_trace = amath.diag(self.decay_rates).dot(self.e_trace) \
                    + self.fifo[-1]
            elif self.insert_to_etrace == "wo_delay":
                self.e_trace = amath.diag(self.decay_rates).dot(self.e_trace) \
                    + f_t
            else:
                raise ValueError("invalid `insert_to_etrace` option.")
        # np.roll implementation is slow.
        self.fifo = amath.roll(self.fifo, 1)
        self.fifo[0] = f_t

        # estimate the next pattern
        if self.n_elg >= 1:
            self.mu = self.variables["b"] \
                + amath.tensordot(self.fifo, self.variables["W"], axes=2)\
                + amath.tensordot(self.e_trace, self.variables["V"], axes=2)

        else:
            self.mu = self.variables["b"] \
                + amath.tensordot(self.fifo, self.variables["W"], axes=2)

        # compare the above numpy-array calculation with naive summation.
        if DEBUG:
            mu_naive = amath.zeros(self.n_anc)
            mu_naive = mu_naive + self.variables["b"]
            for d in xrange(self.delay - 1):
                mu_naive = mu_naive + self.variables["W"][d].dot(self.fifo[d])
            if self.n_elg >= 1:
                for k in xrange(self.n_elg):
                    mu_naive \
                        = mu_naive \
                        + self.variables["V"][k].dot(self.e_trace[k])
            assert amath.allclose(self.mu, mu_naive), "mu has a bug"
        pass

    def _get_delta(self, pattern, loc):
        """
        Get how much we change parameters by learning a given pattern.

        Parameters
        ----------
        pattern: array, shape (n_obs,)
            Pattern used to update the parameters
        loc: array, shape (n_obs, dim)
            Locations of each observation,
            i.e., pattern[i] is observed at loc[i].

        Returns
        -------
        delta: dict
            amount by which the parameters are updated
        """
        _check_pattern_loc_consistency(pattern, loc, self.dim)
        pattern = pattern.ravel()
        n_obs = len(pattern)
        if self.learn_anc_basis:
            self.anc_basis = self.variables["anc_basis"]

        ker_anc_loc = self.ker(self.anc_basis, loc)
        ker_loc_loc = self.ker(loc, loc)
        grad = {}
        s = amath.linalg_solve(
            ker_loc_loc + self.noise_var * amath.identity(n_obs),
            pattern - self.mu.dot(ker_anc_loc))
        grad["b"] = ker_anc_loc.dot(s.reshape(-1, 1)).reshape(-1,)
        grad["W"] \
            = self.fifo[:, amath.newaxis, :] \
            * grad["b"][amath.newaxis, :, amath.newaxis] \
            - self.lmbd_reg["W"] * self.variables["W"]
        if self.n_elg >= 1:
            grad["V"] \
                = self.e_trace[:, amath.newaxis, :] \
                * grad["b"][amath.newaxis, :, amath.newaxis] \
                - self.lmbd_reg["V"] * self.variables["V"]

        if self.learn_anc_basis:
            ker_grad = amath.op.mult_ijk_ij_to_ijk(
                loc[:, amath.newaxis, :]
                - self.variables["anc_basis"][amath.newaxis, :, :],
                ker_anc_loc.transpose())
            s = amath.linalg_solve(ker_loc_loc
                                   + self.noise_var * amath.identity(n_obs),
                                   pattern - self.mu.dot(ker_anc_loc))
            obj_ker_grad = amath.outer(s, self.mu)
            grad["anc_basis"] = amath.op.mult_ijk_ij_to_jk(
                ker_grad, obj_ker_grad)

        # compare the above numpy-array calculation with naive summation.
        if DEBUG:
            grad_W_naive = amath.zeros((self.delay-1, self.n_anc_basis,
                                        self.n_anc_data))
            for d in xrange(self.delay-1):
                grad_W_naive[d] = amath.outer(grad["b"], self.fifo[d])
            grad_W_naive \
                = grad_W_naive - self.lmbd_reg["W"] * self.variables["W"]
            assert amath.allclose(grad_W_naive, grad["W"]), "grad has a bug"
        grad["b"] -= self.lmbd_reg["b"] * self.variables["b"]

        if self.SGD is None:
            delta = {}
            for key in grad:
                delta[key] = self.learning_rate * grad[key]
        else:
            self.SGD.update_state(grad)
            delta = self.SGD.get_delta()
        return delta

    def _update_parameters(self, delta):
        """
        Update the parameters by delta.

        Parameters
        ----------
        delta: dict
            The amount by which the parameters are updated.
        """
        if self.SGD is None:
            for key in delta:
                self.variables[key] = self.variables[key] + delta[key]
        else:
            self.SGD.update_with_L1_regularization(self.variables, delta,
                                                   self.L1_coef)


class FuncLinearDyBM(object):
    ''' A class of a Linear DyBM applied to a functional time series.

    Parameters
    ----------
    dim: int
        Dimension of the feature space.
    anc_points: array, shape (n_anc, dim)
        Anchor points, on which we place LinearDyBM (with ``n_anc`` units).
    delay: int
        Conduction delay of LinearDyBM.
    decay_rates: list, length n_elg
        Decay rates of eligibility traces, each of which is a float in [0, 1).
        ``n_elg`` stands for "the number of eligibility traces".
    noise_var: float, optional
        A variance parameter of a Gaussian distribution that models a noise on
        observation.
    ker_paras: dict, optional
        Parameters of the kernel.
        For example, ``ker_paras = {"ker_type": "rbf", "gamma": 1.0}``.
        For details, see
        `here <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics>`_.
        ``"ker_type"`` is either ``'polynomial'``, ``'laplacian'``, ``'linear'``,
        ``'additive_chi2'``, ``'sigmoid'``, ``'chi2'``, ``'rbf'``, ``'cosine'``, ``'poly'``.
    insert_to_etrace: str, {"w_delay", "wo_delay"},  optional
        ``"w_delay"`` : insert pattern observed d-1 time steps ago into
        eligibility traces.
        ``"wo_delay"`` : insert the latest pattern into eligibility traces.
    SGD: object of SGD.SGD, optional
        Object of a stochastic gradient method.
    learning_rate: float, optional
        Learning rate for SGD.

    Attributes
    ----------
    anc : array, shape (n_anc, dim)
        ``anc[i_anc, :]`` corresponds to the ``i_anc``-th anchor point. fixed.
        Initialized using anc_points.
    l_DyBM : LinearDyBM
        Instance of LinearDyBM class.
    '''

    def __init__(self, dim, anc_points, delay, decay_rates,
                 noise_var=0.1,
                 ker_paras=None, insert_to_etrace="wo_delay",
                 SGD=None, learning_rate=None):
        if not dim == anc_points.shape[1]:
            raise ValueError("inconsistent dimension: "
                             "{}, {}".format(dim, anc_points.shape[1]))
        if insert_to_etrace not in ["w_delay", "wo_delay"]:
            raise ValueError("insert_to_etrace should be either `w_delay` "
                             "or `wo_delay`.")
        self.insert_to_etrace = insert_to_etrace
        self.dim = dim
        self.anc = anc_points
        self.n_anc = self.anc.shape[0]
        if SGD is None:
            SGD = RMSProp()
        self.l_DyBM = LinearDyBM(in_dim=self.n_anc, out_dim=self.n_anc,
                                 delay=delay, decay_rates=decay_rates,
                                 SGD=SGD, insert_to_etrace=insert_to_etrace)
        if learning_rate is not None:
            self.l_DyBM.set_learning_rate(learning_rate)
        self.noise_var = noise_var

        # set kernel function
        if ker_paras is None:  # rbf kernel
            self.ker_paras = {"gamma": 1.0}
            self.ker = partial(amath.kernel_metrics["rbf"], **self.ker_paras)
        else:
            self.ker_paras = ker_paras
            ker_paras = deepcopy(self.ker_paras)
            ker_type = ker_paras.pop("ker_type")
            self.ker = partial(amath.kernel_metrics[ker_type],
                               **ker_paras)

        # to check ker_paras is consistent or not.
        _ = self.ker(amath.zeros((1, 1)), amath.zeros((1, 1)))

        self.ker_anc = self.ker(self.anc, self.anc)
        print(" - condition number of ker_anc + noise = {}".format(
            amath.cond(self.ker_anc
                       + self.noise_var
                       * amath.identity(self.n_anc))))
        self.ker_anc_cho \
            = amath.cho_factor(self.ker_anc
                               + self.noise_var * amath.identity(self.n_anc))
        pass

    def learn_one_step(self, pattern, loc):
        """ Update model parameters using a new observation ``(pattern, loc)``.
        After executing ``learn_one_step()``, it is ready to predict the next pattern using
        ``predict_next()``

        Parameters
        ----------
        pattern: array, shape (n_obs,)
            Pattern used to update the state.
        loc: array, shape (n_obs, dim)
            Locations of each observation,
            i.e., ``pattern[i]`` is observed at ``loc[i]``.
        """
        _check_pattern_loc_consistency(pattern, loc, self.dim)
        pattern = pattern.ravel()
        delta = self._get_delta(pattern, loc)
        self._update_parameters(delta)
        self._update_state(pattern, loc)

    def learn(self, pattern_seq, loc_seq, init_state=False):
        """ This method trains the model by one-pass SGD on the given data.
        After executing this method, the model is ready for predicting the next pattern.

        Parameters
        ----------
        pattern_seq : list
            list of patterns.
        loc_seq : list
            list of locations.
        init_state : bool
            if ``True``, FIFO and eligibility traces are initialized before learning.
        """
        if len(pattern_seq) != len(loc_seq):
            raise ValueError("lengths of pattern_seq and loc_seq are inconsistent.")
        if init_state:
            self.l_DyBM.init_state()
        for each_pattern, each_loc in zip(pattern_seq, loc_seq):
            self.learn_one_step(each_pattern, each_loc)

    def predict_next(self, loc):
        """
        Predict the next pattern in a deterministic manner.

        Parameters
        ----------
        loc: array, shape (n_loc, dim)
            Locations of neurons whose predictions are returned.

        Returns
        -------
        array, shape (n_loc,)
            Predictions.
        """
        _check_loc_consistency(loc, self.dim)
        mu_anc = self.l_DyBM.predict_next()
        c = amath.cho_solve(self.ker_anc_cho, mu_anc).reshape(-1, 1)
        return self.ker(loc, self.anc).dot(c).reshape(-1,)

    def compute_RMSE(self, pattern, loc):
        """
        Compute the RMSE between the prediction and the actual observation.
        Given the model prediction :math:`\mu^{[t+1]}(x)` and the actual
        observation :math:`(X^{[t+1]},f^{[t+1]}(X^{[t+1]}))`, this method
        outputs
        :math:`\dfrac{1}{\sqrt{\mathrm{n\_obs}}}\|\mu^{[t+1]}(X^{[t+1]})
        - f^{[t+1]}(X^{[t+1]})\|_{2}`.

        Parameters
        ----------
        pattern: array, shape (n_obs,)
            Pattern on which we compute RMSE using the prediction for it.
        loc: array, shape (n_obs, dim)
            Locations of each observation of the pattern,
            i.e., ``pattern[i]`` is observed at ``loc[i]``.

        Returns
        -------
        float
            RMSE between pattern and expected pattern.
        """
        _check_pattern_loc_consistency(pattern, loc, self.dim)
        pattern = pattern.ravel()
        n_obs = len(pattern)

        expected_pat = self.predict_next(loc)
        return amath.root_mean_square_err(expected_pat, pattern)

    def get_LL(self, pattern, loc):
        """
        Get the log likelihood of a pattern in LinearDyBM.

        Parameters
        ----------
        pattern: array, shape (n_obs,)
            Pattern on which we compute the log likelihood using the prediction
            for it.
        loc: array, shape (n_obs, dim)
            Locations of each observation of the pattern,
            i.e., ``pattern[i]`` is observed at ``loc[i]``.

        Returns
        -------
        float
            log pdf (probability density function) of the pattern
        """
        _check_pattern_loc_consistency(pattern, loc, self.dim)
        pattern = pattern.ravel()

        f_anc = self._get_f_anc(pattern, loc)
        return l_DyBM.get_LL(f_anc)

    def _get_f_anc(self, pattern, loc):
        """
        Get function values at anchor points, which will be feeded
        to LinearDyBM.

        Parameters
        ----------
        pattern: array, shape (n_obs,)
            Observed pattern.
        loc: array, shape (n_obs, dim)
            Locations of each observation of the pattern,
            i.e., ``pattern[i]`` is observed at ``loc[i]``.

        Returns
        -------
        array, shape (n_anc, dim)
            Estimated pattern at anchor points.
        """
        _check_pattern_loc_consistency(pattern, loc, self.dim)
        pattern = pattern.ravel()
        n_obs = len(pattern)

        ker_anc_loc = self.ker(self.anc, loc)
        ker_loc_loc = self.ker(loc, loc)
        return amath.dot(ker_anc_loc,
                         amath.linalg_solve(
                             ker_loc_loc
                             + self.noise_var
                             * amath.identity(n_obs),
                             pattern).reshape(-1, 1)).reshape(-1,)

    def _update_state(self, pattern, loc):
        """
        Update FIFO and eligibility traces.

        Parameters
        ----------
        pattern: array, shape (n_obs,)
            Pattern used to update the state.
        loc: array, shape (n_obs, dim)
            Locations of each observation,
            i.e., ``pattern[i]`` is observed at ``loc[i]``.
        """
        _check_pattern_loc_consistency(pattern, loc, self.dim)
        pattern = pattern.ravel()
        f_anc = self._get_f_anc(pattern, loc)
        self.l_DyBM._update_state(f_anc)

    def _get_delta(self, pattern, loc):
        """
        Get how much we change parameters to learn the present pattern.

        Parameters
        ----------
        pattern: array, shape (n_obs,)
            Pattern used to update the parameters
        loc: array, shape (n_obs, dim)
            Locations of each observation,
            i.e., ``pattern[i]`` is observed at ``loc[i]``.

        Returns
        -------
        delta: dict
            amount by which the parameters are updated
        """
        _check_pattern_loc_consistency(pattern, loc, self.dim)
        pattern = pattern.ravel()

        f_anc = self._get_f_anc(pattern, loc)
        return self.l_DyBM._get_delta(f_anc)

    def _update_parameters(self, delta):
        """
        update the parameters by delta

        Parameters
        ----------
        delta: dict
            amount by which the parameters are updated
        """
        self.l_DyBM._update_parameters(delta)
