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

from .. import arraymath as amath
from ..base.sgd import RMSProp
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from six.moves import xrange

"""
SARSA temporal-difference learning with Binary DyBM with Uniform Delays

"""

__author__ = "Sakyasingha Dasgupta"


class DYSARSA:

    """ Discrete actions SARSA reinforcement learning with DyBM

    Parameters
    ----------
    n_obs : int
        dimension of actual state vector
    n_actions : int
        dimension of actions
    delay : int, optional
        length of fifo queue plus one
    decay_rates : list, optional
        decay rates of eligibility traces
    SGD: object of SGD.SGD, optional
        Object of a stochastic gradient method.
        If ``None``, we update parameters, fixing the learning rate.
    learnRate: float, optional
        learning rate of SGD
    discount : float
        discounting TD error update
    temperature : float, optional
        hyper-parameter for Boltzmann exploration
    insert_to_etrace : str, {"w_delay", "wo_delay"},  optional
        "w_delay" : insert pattern observed d-1 time steps ago into
        eligibility traces
        "wo_delay" : insert the latest pattern into eligibility traces
    L1 : float, optional
        strength of L1 regularization
    L2 : float, optional
        strength of L2 regularization

    """

    def __init__(self, n_obs, n_actions, delay, decay_rates, discount, SGD=None,
                 learnRate=0.0001, temperature=0.1, insert_to_etrace="wo_delay", L1=0.0, L2=0.00):

        if insert_to_etrace not in ["w_delay", "wo_delay"]:
            raise ValueError("insert_to_etrace should be either `w_delay` "
                             "or `wo_delay`.")

        if delay < 0:
            raise ValueError("delay should not be an integer >=0")

        # observation dimension including action dimension
        self.in_dim = n_obs + n_actions

        self.out_dim = n_actions  # action dimension

        self.delay = delay

        self.decay_rates = decay_rates  # eligibility trace decay rate

        self.n_etrace = len(decay_rates)

        self.discount = discount

        self.epsilon = temperature  # DyBM temperature parameter

        self.variables = {"W": amath.random.normal(0, 1, (self.delay-1, self.in_dim, self.in_dim))
                          / amath.sqrt(self.in_dim),
                          "b": amath.random.normal(0, 1, (self.in_dim, 1)) / amath.sqrt(self.in_dim),
                          "V": amath.random.normal(0, 1, (self.n_etrace, self.in_dim, self.in_dim))
                          / amath.sqrt(self.in_dim)}

        self.L2 = dict()
        self.L1 = dict()

        for key in self.variables:
            self.L1[key] = L1
            self.L2[key] = L2

        if SGD is None:
            SGD = RMSProp()
        self.SGD = SGD.set_shape(self.variables)  # resetting SGD

        self.set_learning_rate(learnRate)

        self.insert_to_etrace = insert_to_etrace

        self.init_state()  # initialize DyBM state

    def init_state(self):
        """ Initialize the state of eligibility traces and FIFO Queue """

        self.e_trace = amath.zeros((self.in_dim, self.n_etrace))  # eligibility trace

        # [x[t-1],x[t-2],...,x[t-D+1]]
        self.fifo = amath.FIFO((max(0, self.delay-1), self.in_dim, 1))

    def _update_state(self, in_pattern):
        """ Update the FIFO Queue with current pattern. Update the eligibility
        trace with current pattern based on the decay rate """

        if len(self.fifo) > 0:
            popped_in_pattern = self.fifo.push(in_pattern)

        if len(self.fifo) == 0:
            popped_in_pattern = in_pattern

        if self.insert_to_etrace == "wo_delay":
            self.e_trace = self.e_trace * self.decay_rates + in_pattern
        elif self.insert_to_etrace == "w_delay":
            self.e_trace = self.e_trace * self.decay_rates + popped_in_pattern
        else:
            print("not implemented.")
            exit(-1)

    def prob_action(self, pattern, epsilon=0.1):
        """ Boltzmann exploration policy for selecting actions """

        num = amath.exp((1. / epsilon) * self.Q_next(pattern))

        denum = 1 + amath.exp((1. / epsilon) * self.Q_next(amath.ones((self.in_dim, 1))))

        return num/denum

    def Q_next(self, pattern):  # E_j(x_j(t) | X(:,t-1))
        """ Q action-value function approximated by the linear energy of
        DyBM """

        bias = self.variables["b"]
        v_weight = self.variables["V"]
        w_weight = self.variables["W"]

        temp = -1*bias

        for i in xrange(self.n_etrace):
                temp -= v_weight[i].dot(self.e_trace[:, i].reshape((self.in_dim, 1)))

        for d in range(self.delay - 1):
                fifo_array = self.fifo.to_array()[d]
                temp -= w_weight[d].dot(fifo_array)

        return amath.multiply(temp, pattern)

    def TD_error(self, reward, futureQ, prevQ):
        """ Update the temporal difference error function  """

        error = reward + self.discount * futureQ - prevQ
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaled_error = scaler.fit_transform(error.reshape(-1, 1))
        return scaled_error.reshape((self.in_dim, 1))

    def Q_error(self, reward, futureQ, prevQ):

        error = ((reward + self.discount * amath.max(futureQ)) - prevQ)**2
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaled_error = scaler.fit_transform(error.reshape(-1, 1)).mean()

        return scaled_error.reshape((self.in_dim, 1))

    def _get_delta(self, obs, error):
        """Update the DySARSA parameter using current observations
        and TD-error"""

        dx = amath.multiply(obs, error)
        gradient = dict()

        if "b" in self.variables:
            gradient["b"] = dx
        if "V" in self.variables:
            gradient["V"] = amath.array([dx * self.e_trace[:, i].transpose()
                                         for i in range(self.n_etrace)])
            # print gradient["V"].shape

        if "W" in self.variables:
            gradient["W"] = amath.array([dx * self.fifo.to_array()[d].transpose()
                                         for d in range(self.delay - 1)])
            # print gradient["W"].shape

        if self.SGD is None:
            delta = {}
            for key in gradient:
                delta[key] = self.learning_rate * gradient[key]
        else:
            self.SGD.update_state(gradient)
            delta = self.SGD.get_delta()

        return delta

    def _update_parameters(self, delta):
        """Update parameters of DySARSA using delta

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
                                                   self.L1)

    def learn_one_step(self, obs, error):
        """ main function call for updating DySARA parameters w.r.t error,
        current observation and FIFO queues and eligibility traces
        """
        delta = self._get_delta(obs, error)
        self._update_parameters(delta)
        self._update_state(obs)

    def set_learning_rate(self, rate):
        """
        Set learning rate of SGD.

        Parameters
        ----------
        rate : float
            Learning rate.
        """
        self.learning_rate = rate
        self.SGD.set_learning_rate(rate)
        pass
