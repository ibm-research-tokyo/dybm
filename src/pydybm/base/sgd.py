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


__author__ = "Takayuki Osogami, Sakyasingha Dasgupta"


from abc import ABCMeta, abstractmethod
from .. import arraymath as amath
from copy import deepcopy
from collections import defaultdict
from functools import partial

import numpy as np
import scipy.linalg as sl


class SGD():
    """
    Abstract stochastic gradient descent

    Parameters
    ----------
    rate : float, optional
        learning rate
    use_func_gradient : bool
        Whether use `func_gradient` in `update_state` method. Defaults to False.
    callback : Callable[[SGD], None], optional
        Callable object for logging/debugging purpose. Called at the end of every `update_state`.
    attr_to_log : str, optional
        Name of attribute to log. This will override the function given by the argument `callback`.

    Attributes
    ----------
    variables : dict
        variables[key] : shape of variable, key
    first : dict
        first[key] : first moment of the gradient of variable, key
    second : dict
        second[key] : second moment of the gradient of variable, key
    alpha : float or dict
        learning rate
        if dict, alpha[key] : learning rate for variable, key
    history : List[Any]
        List of logged variables. Used for logging purpose.
    """

    __metaclass__ = ABCMeta

    def __init__(self, rate=1.0, use_func_gradient=False, callback=None, attr_to_log=None):
        self.set_learning_rate(rate)
        self.use_func_gradient = use_func_gradient
        self.callback = callback
        self.attr_to_log = attr_to_log
        self.history = []
        if self.attr_to_log is not None:

            def logger(obj):
                record = deepcopy(getattr(obj, self.attr_to_log))
                obj.history.append(record)
            self.callback = logger

    def set_shape(self, variables):
        """
        Secondary initialization (set attributes that depends on shapes of variables).

        Parameters
        ----------
        variables : dict
            variables[key] : shape of variable, key

        Returns
        -------
        self : SGD
            self
        """
        self.first = dict()
        self.second = dict()
        for key in variables:
            self.first[key] = amath.zeros(variables[key].shape)
            self.second[key] = amath.zeros(variables[key].shape)
        self.init_state(variables)
        return self

    def init_state(self, variables):
        """
        Initialize state along with shape of variables.
        Nothing is done by default.

        Parameters
        ----------
        variables : dict
            variables[key] : shape of variable, key
        """
        pass

    @abstractmethod
    def get_delta(self):
        """
        getting the vector, delta, to update the associated parameter, a,
        such that a <- a + delta

        Returns
        -------
        delta : dict
            dictionary of the delta vectors of all variables
        """
        pass

    def update_state(self, gradients, params=None, func_gradients=None):
        """
        Wrapper method updating internal state with current parameters and gradient function

        Parameters
        ----------
        gradients : Optional[Dictionary[str, np.ndarray]]
            Dictionary of gradients. It is computed if it is None and func_gradient is set.

        params : Dictionary[str, np.ndarray], optional
            Dictionary of parameters.

        func_gradients : Callable[[Dictionary[str, np.ndarray]], Dictionary[str, np.ndarray]], optioanl
            Function that maps from parameter to gradients.
            If this is not used, `params` is neither used.
            Otherwise, `params` must be set.
        """
        if func_gradients is None:
            if self.use_func_gradient:
                raise ValueError("`func_gradients` must be specified for {}".format(self.__class__.__name__))
            params = None
        else:
            if params is None:
                raise ValueError("`params` must be set if `func_gradient` is used")
            if gradients is None:
                gradients = func_gradients(params)
        self._update_state(gradients, params, func_gradients)
        if self.callback is not None:
            self.callback(self)

    @abstractmethod
    def _update_state(self, gradients, params, func_gradients):
        """
        Virtual method updating internal state with current parameters and gradient function

        Parameters
        ----------
        gradients : Dictionary[str, np.ndarray]
            Dictionary of gradients.
        params : Dictionary[str, np.ndarray]
            Dictionary of parameters.
        func_gradients : Callable[[Dictionary[str, np.ndarray]], Dictionary[str, np.ndarray]]
            Function that maps from parameter to gradients.
        """
        pass

    @abstractmethod
    def get_threshold(self):
        """
        getting threshold for L1 regularization

        Returns
        -------
        threshold : dict
            dictionary of the thresholds for all variables
        """
        pass

    def set_learning_rate(self, rate):
        """
        setting learning rate

        Parameters
        ----------
        rate : float
            learning rate
        """
        self.alpha = rate

    def get_learning_rate(self):
        """
        Getting learning rate
        """
        return self.alpha

    def update_with_L1_regularization(self, variable, delta, strength):
        """
        updating variables with L1 regularization

        Parameters
        ----------
        variable : dict
            dictionary of variables -- to be updated
        delta : dict
            dictionary of amout of update without L1 regularization for all
            variables
        strength : dict
            dictionary of float strength of regularization for all variables
        """
        th = None
        for key in delta:
            if strength[key] == 0.0:
                variable[key] += delta[key]
            else:
                if th is None:
                    th = self.get_threshold()
                variable[key] = amath.op.sgd_L1_regularization(variable[key],
                                                               delta[key],
                                                               strength[key],
                                                               th[key])

    def apply_L2_regularization(self, gradient, variable, strength):
        """
        applying L2 regularization to the gradient

        Parameters
        ----------
        gradient : dict
            dictionary of gradients for all variables -- to be updated
        variable : dict
            dictionary of variables
        strength : dict
            dictionary of float strength of regularization for all variables
        """
        for key in gradient:
            if key not in strength:
                continue
            if strength[key] == 0.0:
                continue
            if key not in variable:
                continue
            gradient[key] = amath.op.sgd_L2_regularization(gradient[key],
                                                           strength[key],
                                                           variable[key])

    def slice_history(self, key):
        """
        Slice history of variables, assuming that variable[key] makes sense.

        Parameters
        ----------
        key : Hashable
            key of slice

        Returns
        -------
        sliced : List[Any]
            Satisfies that `sliced[t] == self.history[t][key]`
        """
        return [h[key] for h in self.history]


class RMSProp(SGD):
    """
    RMSProp

    Parameters
    ----------
    variables : dict
        variables[key] : shape of variable, key
    gamma : float, optional
          discount factor
    rate : float, optional
        learning rate

    Attributes
    ----------
    first : dict
        first[key] : first moment of the gradient of variable, key
    second : dict
        second[key] : second moment of the gradient of variable, key
    alpha : float
        learning rate
    gamma : float
        discount factor
    """

    def __init__(self, rate=0.001, gamma=0.9, delta=1e-8):
        self.step = 0
        self.delta = delta
        self.gamma = gamma
        SGD.__init__(self, rate)

    def get_delta(self):
        """
        getting the vector, delta, to update the associated parameter, a,
        such that a <- a + delta

        Returns
        -------
        delta : dict
            dictionary of the delta vectors for all variables
        """
        delta = dict()
        for key in self.first:
            delta[key] = amath.op.rmsprop_get_delta(self.alpha,
                                                    self.first[key],
                                                    self.second[key],
                                                    self.delta)

        return delta

    def _update_state(self, gradients, params, func_gradients):
        """
        Virtual method updating internal state with current parameters and gradient function

        Parameters
        ----------
        gradients : Dictionary[str, np.ndarray]
            Dictionary of gradients.
        params : Dictionary[str, np.ndarray]
            Dictionary of parameters.
        func_gradients : Callable[[Dictionary[str, np.ndarray]], Dictionary[str, np.ndarray]]
            Function that maps from parameter to gradients.
        """
        g = gradients
        for key in g:
            self.first[key] = g[key]
            self.second[key] = amath.op.rmsprop_update_state(self.gamma,
                                                             self.second[key],
                                                             g[key])

    def get_threshold(self):
        """
        getting threshold for L1 regularization

        Returns
        ----------
        threshold : dict
            dictionary of the thresholds for all variables
        """
        th = dict()
        for key in self.first:
            th[key] = amath.op.rmsprop_get_threshold(self.alpha,
                                                     self.second[key],
                                                     self.delta)
        return th


def timer():

    """
    abstract function used by noisy RMSProp to update the amount of time steps for noisy gradient calculations

    """

    try:
        timer.counter += 1
    except AttributeError:
        timer.counter = 1
    return timer.counter


class NoisyRMSProp(SGD):
    """
    RMSProp with annealing noise added to gradient

    Parameters
    ----------
    variables : dict
        variables[key] : shape of variable, key
    gamma : float, optional
          discount factor
    rate : float, optional
        learning rate

    Attributes
    ----------
    first : dict
        first[key] : first moment of the gradient of variable, key
    second : dict
        second[key] : second moment of the gradient of variable, key
    alpha : float
        learning rate
    gamma : float
        discount factor
    """

    def __init__(self, rate=0.001, gamma=0.9, delta=1e-8, gau=0.55, eta=0.1):
        self.step = 0
        self.delta = delta
        self.gamma = gamma
        self.gau = gau
        self.eta = eta
        SGD.__init__(self, rate)

    def get_delta(self):
        """
        getting the vector, delta, to update the associated parameter, a,
        such that a <- a + delta

        Returns
        -------
        delta : dict
            dictionary of the delta vectors for all variables
        """

        times = timer()
        sigma = self.eta / (1 + times) ** self.gau

        delta = dict()

        for key in self.first:
            noise = np.random.normal(0, sigma)
            delta[key] = amath.op.noisyrmsprop_get_delta(self.alpha,
                                                         self.first[key],
                                                         self.second[key],
                                                         self.delta, noise)

        return delta

    def _update_state(self, gradients, params, func_gradients):
        """
        Virtual method updating internal state with current parameters and gradient function

        Parameters
        ----------
        gradients : Dictionary[str, np.ndarray]
            Dictionary of gradients.
        params : Dictionary[str, np.ndarray]
            Dictionary of parameters.
        func_gradients : Callable[[Dictionary[str, np.ndarray]], Dictionary[str, np.ndarray]]
            Function that maps from parameter to gradients.
        """
        g = gradients
        for key in g:
            self.first[key] = g[key]
            self.second[key] = amath.op.noisyrmsprop_update_state(self.gamma,
                                                                  self.second[key],
                                                                  g[key])

    def get_threshold(self):
        """
        getting threshold for L1 regularization

        Returns
        ----------
        threshold : dict
            dictionary of the thresholds for all variables
        """
        th = dict()
        for key in self.first:
            th[key] = amath.op.noisyrmsprop_get_threshold(self.alpha,
                                                          self.second[key],
                                                          self.delta)
        return th


class AdaGrad(SGD):
    """
    AdaGrad

    Parameters
    ----------
    variables : dict
        variables[key] : shape of variable, key
    rate : float, optional
        learning rate

    Attributes
    ----------
    first : dict
        first[key] : first moment of the gradient of variable, key
    second : dict
        second[key] : second moment of the gradient of variable, key
    alpha : float
        learning rate
    """

    def __init__(self, rate=1.0, delta=1e-6):
        self.step = 0
        self.delta = delta
        SGD.__init__(self, rate)

    def get_delta(self):
        """
        getting the vector, delta, to update the associated parameter, a,
        such that a <- a + delta

        Returns
        -------
        delta : dict
            dictionary of the delta vectors for all variables
        """
        delta = dict()
        for key in self.first:
            delta[key] = amath.op.adagrad_get_delta(self.alpha,
                                                    self.first[key],
                                                    self.second[key],
                                                    self.delta)

        return delta

    def _update_state(self, gradients, params, func_gradients):
        """
        Virtual method updating internal state with current parameters and gradient function

        Parameters
        ----------
        gradients : Dictionary[str, np.ndarray]
            Dictionary of gradients.
        params : Dictionary[str, np.ndarray]
            Dictionary of parameters.
        func_gradients : Callable[[Dictionary[str, np.ndarray]], Dictionary[str, np.ndarray]]
            Function that maps from parameter to gradients.
        """
        g = gradients
        for key in g:
            self.first[key] = g[key]
            self.second[key] = amath.op.adagrad_update_state(self.second[key], g[key])

    def get_threshold(self):
        """
        getting threshold for L1 regularization

        Returns
        -------
        threshold : dict
            dictionary of the thresholds for all variables
        """
        th = dict()
        for key in self.first:
            th[key] = amath.op.adagrad_get_threshold(self.alpha,
                                                     self.second[key],
                                                     self.delta)

        return th


class AdaGradPlus(AdaGrad):
    """
    AdaGrad with first moment updated like ADAM

    Parameters
    ----------
    variables : dict
        variables[key] : shape of variable, key
    rate : float, optional
        learning rate

    Attributes
    ----------
    first : dict
        first[key] : first moment of the gradient of variable, key
    second : dict
        second[key] : second moment of the gradient of variable, key
    alpha : float
        learning rate
    beta : float
    """

    def __init__(self, rate=1.0, delta=1e-6, beta=0.0):
        self.beta = beta
        AdaGrad.__init__(self, rate, delta)

    def _update_state(self, gradients, params, func_gradients):
        """
        Virtual method updating internal state with current parameters and gradient function

        Parameters
        ----------
        gradients : Dictionary[str, np.ndarray]
            Dictionary of gradients.
        params : Dictionary[str, np.ndarray]
            Dictionary of parameters.
        func_gradients : Callable[[Dictionary[str, np.ndarray]], Dictionary[str, np.ndarray]]
            Function that maps from parameter to gradients.
        """
        g = gradients
        for key in g:
            self.first[key], self.second[key] \
                = amath.op.adagradplus_update_state(self.first[key],
                                                    self.second[key],
                                                    self.beta, g[key])

    def set_beta(self, beta):
        """
        Setting the value of beta parameter

        Parameters
        ----------
        alpha : float
            value of beta
        """
        self.beta = beta


class ADAM(SGD):
    """
    ADAM

    Parameters
    ----------
    variables : dict
        variables[key] : shape of variable, key
    alpha : float, optional
        parameter of ADAM (learning rate)
    beta : float, optional
        parameter of ADAM
    gamma : float, optional
        parameter of ADAM
    epsilon : float, optional
        parameter of ADAM

    Attributes
    ----------
    first : dict
        first[key] : first moment of the gradient of variable, key
    second : dict
        second[key] : second moment of the gradient of variable, key
    alpha : float
        alpha
    beta : float
        beta
    gamma : float
        gamma
    epsilon : float
        epsilon
    """

    def __init__(self, alpha=0.001, beta=0.9, gamma=0.999,
                 epsilon=1e-8):
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.step = 0

        SGD.__init__(self, alpha)

    def get_delta(self):
        """
        getting the vector, delta, to update the associated parameter, a,
        such that a <- a + delta

        Returns
        -------
        delta : dict
            dictionary of the delta vectors for all variables
        """
        delta = dict()
        for key in self.first:
            delta[key] = amath.op.adam_get_delta(self.first[key],
                                                 self.second[key],
                                                 self.alpha,
                                                 self.beta,
                                                 self.gamma,
                                                 self.epsilon,
                                                 self.step)
        return delta

    def _update_state(self, gradients, params, func_gradients):
        """
        Virtual method updating internal state with current parameters and gradient function

        Parameters
        ----------
        gradients : Dictionary[str, np.ndarray]
            Dictionary of gradients.
        params : Dictionary[str, np.ndarray]
            Dictionary of parameters.
        func_gradients : Callable[[Dictionary[str, np.ndarray]], Dictionary[str, np.ndarray]]
            Function that maps from parameter to gradients.
        """
        g = gradients
        for key in g:
            self.first[key], self.second[key] \
                = amath.op.adam_update_state(self.first[key],
                                             self.second[key],
                                             self.beta,
                                             self.gamma,
                                             g[key])
        self.step += 1

    def get_threshold(self):
        """
        getting threshold for L1 regularization

        Returns
        -------
        threshold : dict
            dictionary of the thresholds for all variables
        """
        th = dict()
        for key in self.first:
            th[key] = amath.op.adam_get_threshold(self.second[key],
                                                  self.alpha,
                                                  self.beta,
                                                  self.gamma,
                                                  self.epsilon,
                                                  self.step)
        return th

    def set_alpha(self, alpha):
        self.set_learning_rate(alpha)

    def set_beta(self, beta):
        self.beta = beta

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


class SGDN_HD(SGD):
    """
    SGD + Nesterov momentum + Hypergradient Descent

    Parameters
    ----------
    variables : dict
        variables[key] : shape of variable, key
    initial_rate : float, optional
        initial learning rate
    hyper_rate : float, optional
        hyper learning rate
    momentum : 0 <= float < 1, optional
        Nesterov momentum
    vectorize : bool
        whether vectorize the step size
    multiplicative : bool
        whether update stepsize in a multiplicative manner (otherwise additive)
    decay_normalization : float, optinal
        decaying factor of normalizing weight of hypergradient
    rate_min : float, optional
        minimum learning rate, for stability
    rate_max : float, optinal
        maximum learning rate, for stability
    callback : function, optional
        If not None, it is called at the end of every call of `update_state`
    record_attr : str, optional
        Set callback to record the attribute designated by `record_attr` with `self.history`.
        In this case `self.history[key]` gives a list of attribute in the chronological order.
        Assume that the attribute is instance of dictionary.

    Attributes
    ----------
    first : dict
        first[key] : previous gradient
    second : dict
        second[key] : exponential average of second moments
    eta : float
        learning rate
    beta : float
        hyper learning rate
    mu : float
        momentum

    veolocity : dict
        velocity[key] : velocity of the gradient
    """

    def __init__(self, initial_rate=1e-3, hyper_rate=1e-3, momentum=0.0,
                 vectorize=False, multiplicative=False, decay_normalization=0.99,
                 rate_min=1e-10, rate_max=0.1, callback=None, attr_to_log=None):
        self.beta = hyper_rate
        self.mu = momentum
        self.vectorize = vectorize
        self.multiplicative = multiplicative
        self.decay_normalization = decay_normalization

        self.rate_min = rate_min
        self.rate_max = rate_max
        self.callback = callback

        SGD.__init__(self, rate=initial_rate, callback=callback, attr_to_log=attr_to_log)
        self.eta = dict()
        self.velocity = dict()
        self.second = dict()
        self._norm_second = 0

    def init_state(self, variables):
        """
        Initialize state
        Parameters
        ----------
        variables : Dictionary[str, np.ndarray]
            shape of variables
        """
        self.eta = {key: self.alpha for key in variables}
        self.velocity = {key: np.zeros_like(val) for key, val in variables.items()}
        self.second = {key: np.zeros_like(val) for key, val in variables.items()}
        self._norm_second = 0

    def get_delta(self):
        """
        getting the vector, delta, to update the associated parameter, a,
        such that a <- a + delta

        Returns
        -------
        delta : dict
            dictionary of the delta vectors for all variables
        """
        delta = dict()
        for key in self.first:
            delta[key] = self.eta[key] * (self.first[key] + self.mu * self.velocity[key])
        return delta

    def _update_state(self, gradients, params, func_gradients):
        """
        Virtual method updating internal state with current parameters and gradient function

        Parameters
        ----------
        gradients : Dictionary[str, np.ndarray]
            Dictionary of gradients.
        params : Dictionary[str, np.ndarray]
            Dictionary of parameters.
        func_gradients : Callable[[Dictionary[str, np.ndarray]], Dictionary[str, np.ndarray]]
            Function that maps from parameter to gradients.
        """
        g = gradients
        gamma = self.decay_normalization
        # choose update rule
        if self.multiplicative:
            updater = (lambda a, da: np.clip(a * np.exp(da), self.rate_min, self.rate_max))
        else:
            updater = (lambda a, da: np.clip(a + da, self.rate_min, self.rate_max))
        # compute hypergradient and normalizer
        self._h = {key: g[key] * (-self.first[key] - self.mu * self.velocity[key]) for key in g}
        if self.decay_normalization == 0:  # for sign-based update
            self._norm_second = 1
            self.second = {key: g[key] * self.first[key] for key in g}
        else:
            self._norm_second = gamma * self._norm_second + (1 - gamma)
            self.second = {key: gamma * self.second[key] + (1 - gamma) * g[key] ** 2 for key in g}
        normalizer = {key: val / self._norm_second for key, val in self.second.items()}
        if not self.vectorize:
            sum_h = (
                sum(np.sum(val) for val in self._h.values())
            )
            sum_normalizer = (
                sum(np.sum(val) for val in normalizer.values())
            )
            self._h = {key: sum_h for key in self._h}
            normalizer = {key: sum_normalizer for key in g}
        # finialize
        normalizer = {key: abs(normalizer[key]) for key in g}
        # update step size
        self.eta = {
            key: updater(self.eta[key], -self.beta * self._h[key] / (normalizer[key] + 1e-15))
            for key in g}
        # update other states
        self.first = g.copy()
        self.velocity = {key: self.mu * self.velocity[key] + g[key] for key in g}

    def get_threshold(self):
        """
        getting threshold for L1 regularization (curretly no regularization)

        Returns
        -------
        threshold : dict
            dictionary of the thresholds for all variables
        """
        return defaultdict(lambda: 0)


class Almeida(SGDN_HD):
    """
    SGD presentd by Almeida (1998) "Parameter Adaptation in Stochastic Optimzation".
    See SGDN_HD for the detail.
    """

    def __init__(self, variables, initial_rate=1e-3, hyper_rate=1e-2,
                 momentum=0.0, vectorize=True, decay_normalization=0,
                 rate_min=1e-10, rate_max=0.1, callback=None, attr_to_log=None):
        SGDN_HD.__init__(
            self, initial_rate=initial_rate, multiplicative=True, hyper_rate=hyper_rate,
            momentum=momentum, vectorize=vectorize, decay_normalization=decay_normalization,
            rate_min=rate_min, rate_max=rate_max, callback=callback, attr_to_log=attr_to_log)


class BBProp:

    """
    Estimator of Hessian for vSGD.
    See vSGD for typical usage.

    Parameters
    ----------
    eps : float
        Small constant for numerical stability
    rng : np.random.RandomState
        PRNG

    Attributes
    ----------
    psi : Dictionary[str, np.ndarray]
        psi[key] is estimate of primary eigenvector(s) of Hessian
    """

    def __init__(self, eps=1e-8, rng=None):
        self.eps = eps
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng

    def set_shape(self, variables):
        """
        Set shape of variables.

        Parameters
        ----------
        variables : Dictionary[str, Tuple[int, ...]]
            Shape of parameters
        Returns
        -------
        self : BBProp
            self
        """
        self.psi = {k: self.rng.normal(size=variables[k].shape) for k in variables}
        return self

    def set_mode(self, mode):
        """
        Set mode of estimation.
        Parameters
        ----------
        mode : str
            Mode of estimation. Either of 'local', 'block' and 'global'.
        Returns
        -------
        self : BBProp
            self
        """
        assert mode in ['local', 'block', 'global']
        self.mode = mode
        return self

    def estimate_hessian(self, gradients, params, func_gradients):
        """
        Estimate Hessian.

        Parameters
        ----------
        gradients : Dictionary[str, np.ndarray]
            Gradients.
        params : Dictionary[str, np.ndarray]
            Parameters.
        func_gradients : Callable[[Dictionary[str, np.ndarray]], Dictionary[str, np.ndarray]]
            Funciton that maps from params to gradients.

        Returns
        -------
        h_max : Dictionary[str, np.ndarray]
            Estimates of maximum eigenvalue of Hessian.
        """
        params1 = {k: params[k] + self.eps * self.psi[k] for k in gradients}
        gradients1 = func_gradients(params1)
        h_psi = {
            k: (gradients1[k] - gradients[k]) / self.eps *
            (1 + self.rng.normal(scale=self.eps, size=gradients[k].shape))  # add small noise for stability
            for k in gradients}
        h_max = dict()
        if self.mode == 'local':
            h_max = {k: abs(h_psi[k]) for k in gradients}
        elif self.mode == 'block':
            h_max = {k: sl.norm(h_psi[k]) for k in gradients}
        elif self.mode == 'global':
            h_max_value = np.sum([np.sum(v ** 2) for v in gradients.values()])
            h_max = {k: h_max_value for k in gradients}
        self.psi = {k: h_psi[k] / h_max[k] for k in gradients}
        return h_max


class vSGD(SGD):
    """
    vSGD proposed by Schaul, Zhang & LeCun (2013) "No More Pesky Learning Rates"

    Parameters
    ----------
    variables : dict
        variables[key]: variable to be optimized
    hessian : dict, optional
        hessian[key]: initial estimate of hessian of objective, default to ones.
    bbprop : BBProp
        Estiamtor of Hessian
    initial_memory : float, optional
        minimum size of memory (>1)
    mode : str, optional, ['local', 'block', 'global']
        granularity of adaptivity
    callback : function, optional
        called for each time `update_state` is invoked
    attr_to_log : str, optional
        Set callback to record the attribute designated by `record_attr` with `self.history`.
        In this case `self.history[key]` gives a list of attribute in the chronological order.
        Assume that the attribute is instance of dictionary.
    eps : float
        small constant for numerical stability

    Attributes
    ----------
    eta : dict
        eta[key] : learning rate
    history : Any
        history of values of an attribute (utility for debugging purpose)
    """

    def __init__(self, hessian=None, bbprop=None, initial_memory=2, mode='local',
                 callback=None, attr_to_log=None, eps=1e-8):
        if mode != 'local':
            raise NotImplementedError
        self.mode = mode
        if hessian is None:
            hessian = defaultdict(lambda: 1)
        self.hessian = deepcopy(hessian)
        if bbprop is None:
            bbprop = BBProp()
        self.bbprop = bbprop
        self.bbprop.set_mode(mode)
        self.eps = eps
        self.initial_memory = initial_memory

        super(vSGD, self).__init__(use_func_gradient=True, callback=callback, attr_to_log=attr_to_log)

    def init_state(self, variables):
        """
        Initialize state
        Parameters
        ----------
        variables : Dictionary[str, np.ndarray]
            shape of variables
        """
        eps = self.eps
        self.first = {k: v + eps for k, v in self.first.items()}
        self.second = {k: v + eps for k, v in self.second.items()}
        self.gradient = deepcopy(self.first)
        self.eta = deepcopy(self.first)
        self.bbprop.set_shape(variables)
        self.memory_size = {k: self.initial_memory * np.ones_like(v) for k, v in self.first.items()}
        self.minimum_memory = {k: self.initial_memory * np.ones_like(v) for k, v in self.first.items()}

    def get_delta(self):
        """
        getting the vector, delta, to update the associated parameter, a,
        such that a <- a + delta

        Returns
        -------
        delta : dict
            dictionary of the delta vectors of all variables
        """
        return {
            k: self.first[k] ** 2 / (self.hessian[k] * self.second[k]) * self.gradient[k]
            for k in self.gradient
        }

    def _update_state(self, gradients, params, func_gradients):
        """
        Virtual method updating internal state with current parameters and gradient function

        Parameters
        ----------
        gradients : Dictionary[str, np.ndarray]
            Dictionary of gradients.
        params : Dictionary[str, np.ndarray]
            Dictionary of parameters.
        func_gradients : Callable[[Dictionary[str, np.ndarray]], Dictionary[str, np.ndarray]]
            Function that maps from parameter to gradients.
        """
        # Update first and second moments
        g = gradients
        decay = {k: 1 / v for k, v in self.memory_size.items()}
        self.first = {k: (1 - decay[k]) * self.first[k] + decay[k] * g[k] for k in g}
        self.second = {k: (1 - decay[k]) * self.second[k] + decay[k] * g[k] ** 2 for k in g}
        # Update Hessian
        hessian = self.bbprop.estimate_hessian(gradients, params, func_gradients)
        self.hessian = {
            k: (1 - decay[k]) * self.hessian[k] + decay[k] * hessian[k]
            for k in g
        }

        # Update step size and memory size
        self.gradient = g
        self.eta = {
            k: self.first[k] ** 2 / (self.hessian[k] * self.second[k])
            for k in g
        }
        self.memory_size = {
            k: np.clip(
                (1 - self.first[k] ** 2 / self.second[k]) * self.memory_size[k] + 1,
                self.minimum_memory[k], np.inf
            ) for k in g
        }

    def get_threshold(self):
        """
        getting threshold for L1 regularization

        Returns
        -------
        threshold : dict
            dictionary of the thresholds for all variables
        """
        return defaultdict(lambda: 0)

