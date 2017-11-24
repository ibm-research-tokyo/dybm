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

"""Implementation of Gaussian DyBMs whose parameters are learned in batch manner.

"""

__author__ = "Rudy Raymond"


import numpy as np
from copy import deepcopy
from six.moves import xrange

from .. import arraymath as amath
from ..time_series.rnn_gaussian_dybm import GaussianDyBM
from ..time_series.batch_dybm import BatchLinearDyBM
from ..base.generator import SequenceGenerator
from ..base.sgd import AdaGrad


class BatchGaussianDyBM(GaussianDyBM):

    """
    GaussianDyBM with batch learning over past input sequence

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
    batch_method: string, optional
        Choose from "Ridge", "Lasso", or "MultiTaskLasso". Default is Ridge
    positive: boolean, optional
        If true and batch_method="Lasso", the coefficients are all positive
        values

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
    fifo : deque
        FIFO queue storing len_fifo in_patterns, each in_pattern has shape (in_dim,).
    e_trace : array, shape (n_etrace, in_dim)
        e_trace[k, :] corresponds to the k-th eligibility trace.
    """

    def __init__(self, in_dim, out_dim=None, delay=2, decay_rates=[],
                 SGD=None, L1=0., L2=0., insert_to_etrace="wo_delay",
                 batch_method="Ridge", positive=False):
        self.batch_method = batch_method
        self._positive = positive
        GaussianDyBM.__init__(self, in_dim, out_dim=out_dim, delay=delay,
                              decay_rates=decay_rates, SGD=SGD, L1=L1, L2=L2,
                              insert_to_etrace=insert_to_etrace)

    def init_state(self, in_seq=None, out_seq=None):
        """
        Initializing FIFO queue and eligibility traces.
        Weight matrices of GaussianDyBM are initialized by batch-learning on
        in_seq and out_seq, if in_seq is not None.
        Otherwise, if in_seq is None it is the same as GaussianDyBM.

        Parameters
        ----------
        in_seq : optional, list of numpy or arraymath array each of shape (in_dim, )
            input sequence
        out_seq : optional, list of numpy or arraymath array each of shape (out_dim, )
            output sequence

        """

        GaussianDyBM.init_state(self)
        if in_seq is not None:
            if out_seq is None:
                out_seq = in_seq
            self._learn_batch(in_seq, out_seq)

    def fit(self, in_seq, out_seq=None, createDyBM=True):
        """
        Fit GaussianDyBM with in_seq and out_seq and set FIFO and
        the eligibility trace

        Parameters
        ----------
        in_seq: list of arraymath array each of shape(in_dim, )
            input sequence
        out_seq: optional, list of numpy or arraymath array each of shape
            (out_dim, ) output sequence
        createDyBM: optional, boolean
            if True, return GaussianDyBM fitted to in_seq and out_seq

        Returns
        -------
        GaussianDyBM fitted to in_seq and out_seq if createDyBM is True

        """

        self.learn_batch(in_seq, out_seq)
        for in_pattern in in_seq:
            self._update_state(in_pattern)

        if createDyBM:
            return super(BatchGaussianDyBM, self)

    def predict(self, in_seq):
        """
        Predict using GaussianDyBM without updating its parameters.
        Use learn_one_step() to learn with parameter updates

        Parameters
        ----------
        in_seq: list of arraymath array each of shape(in_dim, )
            input sequence

        Returns
        -------
        out_seq : sequence of output
            generator of output sequence

        """

        answer = []
        for in_pattern in in_seq:
            self._update_state(in_pattern)
            answer.append(self.predict_next())
        return SequenceGenerator(answer)

    def learn_batch(self, in_seq, out_seq=None):
        """
        Initialize weight matrices of GaussianDyBM by batch-learning

        Parameters
        ----------
        in_seq : list of numpy or arraymath array each of shape (in_dim, )
            input sequence
        out_seq : optional, list of numpy or arraymath array each of shape
            (out_dim, ) output sequence

        """

        if out_seq is None:
            out_seq = in_seq
        self._learn_batch(in_seq, out_seq)

    def _learn_batch(self, in_seq, out_seq):
        """
        private method to perform batch learning

        """

        in_seq = [amath.array(i) for i in in_seq]
        out_seq = [amath.to_numpy(o) for o in out_seq]

        #  get mean and variances of out_seq
        means = np.zeros(self.out_dim)
        stdevs = np.zeros(self.out_dim)
        for i, each in enumerate(zip(*out_seq)):
            means[i] = np.mean(each)
            stdevs[i] = np.std(each)
        #  normalize out_seq
        normalized_out_seq = deepcopy(out_seq)
        for j in xrange(len(normalized_out_seq)):
            normalized_out_seq[j] = (normalized_out_seq[j] - means)/stdevs

        #  perform BatchLinearDyBM on normalized out_seq
        linDyBM = BatchLinearDyBM(self.in_dim, self.out_dim, delay=self.len_fifo+1,
                                  decay_rates=self.decay_rates, SGD=AdaGrad(),
                                  L1=self.L1["V"], L2=self.L2["V"],
                                  use_bias=True, sigma=0,
                                  insert_to_etrace=self.insert_to_etrace,
                                  esn=None, batch_method=self.batch_method,
                                  positive=self._positive)
        linDyBM.init_state(in_seq, normalized_out_seq)  # initialize parameters
        normalized_pred = linDyBM.predict(in_seq).to_list()  # get parameters

        #  get mean squared errors
        sqerrs = np.zeros(self.out_dim)
        for t in xrange(len(normalized_pred)):
            y1, y2 = normalized_pred[t], normalized_out_seq[t]
            for i in xrange(self.out_dim):
                sqerrs[i] += (y1[i] - y2[i])**2
        T = len(normalized_pred)
        for i in xrange(self.out_dim):
            sqerrs[i] /= T

        #  assign standard deviations
        for i in xrange(self.out_dim):
            self.variables["s"][0][i] = np.sqrt(sqerrs[i]) * stdevs[i]

        #  assign parameters of GaussianDyBM from LinearDyBM
        Sigma = amath.diag(amath.array(stdevs))
        self.variables["b"] = \
            amath.dot(Sigma, linDyBM.variables["b"]) + amath.array(means)
        for l in range(self.len_fifo):
            self.variables["W"][l] = amath.dot(linDyBM.variables["W"][l], Sigma)

        for k in range(self.n_etrace):
            self.variables["V"][k] = amath.dot(linDyBM.variables["V"][k], Sigma)
