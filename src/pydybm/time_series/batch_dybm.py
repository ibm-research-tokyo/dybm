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

"""Implementation of DyBMs whose parameters are learned in batch manner.

"""

__author__ = "Rudy Raymond, Kun Zhao"


import numpy as np
import sklearn.linear_model
from six.moves import xrange

from .. import arraymath as amath
from ..time_series.dybm import LinearDyBM, MultiTargetLinearDyBM
from ..base.sgd import AdaGrad
from ..base.generator import SequenceGenerator
from ..time_series.time_series_model import StochasticTimeSeriesModel


DEBUG = False


class BatchLinearDyBM(LinearDyBM):
    """
    LinearDyBM with batch training

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
    use_bias : boolean, optional
        whether to use bias parameters
    sigma : float, optional
        standard deviation of initial values of weight parameters
    insert_to_etrace : str, {"w_delay", "wo_delay"},  optional
        "w_delay" : insert pattern observed d-1 time steps ago into
        eligibility traces
        "wo_delay" : insert the latest pattern into eligibility traces
    esn : ESN, optional
        Echo state network
    batch_method: string, optional
        Choose from "Ridge", "Lasso", or "MultiTaskLasso". Default is Ridge
    positive: boolean, optional
        If true and batch_method="Lasso", the coefficients are all positive
        values
    learn_beginning: boolean. optional
        If true, the model will perform learning from first step.
        If false, the model will perform learning from delay - 1 step.

    Attributes
    ----------
    decay_rates : array, shape (n_etrace, 1)
        decay rates of eligibility traces
    e_trace : array, shape (n_etrace, in_dim)
        e_trace[k, :] corresponds to the k-th eligibility trace.
    esn : ESN
        esn
    fifo : deque
        FIFO queue storing len_fifo in_patterns, each in_pattern has shape (in_dim,).
    insert_to_etrace : str
        insert_to_etrace
    n_etrace : int
        the number of eligibility traces
    len_fifo : int
        the length of FIFO queues (delay - 1)
    L1 : dict
        dictionary of the strength of L1 regularization
    L1[x] : float
        strength of L1 regularization for variable x for x in ["b","V","W"]
    L2 : dict
        dictionary of the strength of L2 regularization
    L2[x] : float
        strength of L2 regularization for variable x for x in ["b","V","W"]
    in_dim : int
        in_dim
    out_dim : int
        out_dim
    SGD : SGD
        Optimizer used in the stochastic gradient method
    variables : dict
        dictionary of model parameters
    variables["W"] : array, shape (len_fifo, in_dim, out_dim)
        variables["W"][l] corresponds to the weight from the input observed
        at time step t - l - 1 to the mean at time step t (current time).
    variables["b"] : array, shape (out_dim,)
        variables["b"] corresponds to the bias to out_pattern.
    variables["V"] : array, shape (n_etrace, in_dim, out_dim)
        variables["V"][k] corresponds to the weight from the k-th eligibility
        trace to the mean.
    """

    def __init__(self, in_dim, out_dim=None, delay=2, decay_rates=[0.5],
                 SGD=None, L1=0, L2=0, use_bias=True, sigma=0,
                 insert_to_etrace="wo_delay", esn=None,
                 batch_method=None, positive=False, learn_beginning=True):
        # assert esn is None, \
        #     "Batch method for ESN is not supported yet."
        # initialize base class
        LinearDyBM.__init__(self, in_dim, out_dim=out_dim, delay=delay,
                            decay_rates=decay_rates, SGD=SGD, L1=L1, L2=L2,
                            use_bias=use_bias, sigma=sigma,
                            insert_to_etrace=insert_to_etrace, esn=esn)
        # initialize batch parameters
        self.batch_method = batch_method
        self._positive = positive
        self._use_bias = use_bias
        self._L1 = L1
        self._L2 = L2
        self._sigma = sigma
        self.learn_beginning = learn_beginning

    def init_state(self, in_seq=None, out_seq=None):
        """
        Initializing FIFO queue and eligibility traces.
        Weight matrices of DyBM are initialized by batch-learning on in_seq
        and out_seq, if in_seq is not None.
        Otherwise, if in_seq is None it is the same as LinearDyBM.

        Parameters
        ----------
        in_seq : optional, list of numpy or arraymath array each of shape
            (in_dim, ) input sequence
        out_seq : optional, list of numpy or arraymath array each of shape
            (out_dim, ) output sequence

        """

        LinearDyBM.init_state(self)
        if in_seq is not None:
            if out_seq is None:
                out_seq = in_seq
            self._learn_batch(in_seq, out_seq)

    def fit(self, in_seq, out_seq=None, createDyBM=True):
        """
        Fit LinearDyBM with in_seq and out_seq and set FIFO and
        the eligibility trace

        Parameters
        ----------
        in_seq: list of arraymath array each of shape(in_dim, )
            input sequence
        out_seq: optional, list of numpy or arraymath array each of shape
            (out_dim, ) output sequence
        createDyBM: optional, boolean
            if True, return LinearDyBM fitted to in_seq and out_seq

        Returns
        -------
        LinearDyBM fitted to in_seq and out_seq if createDyBM is True

        """

        self.learn_batch(in_seq, out_seq)
        for in_pattern in in_seq:
            self._update_state(in_pattern)

        if createDyBM:
            return super(BatchLinearDyBM, self)

    def fit_multi_seqs(self, in_seq, out_seq=None, createDyBM=True):
        """
        Fit multi-sequence with same or different length.

        Parameters
        ----------
        in_seq: list of (list of arraymath array) each of shape(in_dim, )
            input sequences
        out_seq: optional, list of (list of numpy or arraymath array) each of shape
            (out_dim, ) output sequences
        createDyBM: optional, boolean
            if True, return LinearDyBM fitted to in_seq and out_seq

        Returns
        -------
        LinearDyBM fitted to in_seq and out_seq if createDyBM is True

        """

        if out_seq is None:
            out_seq = in_seq

        self._learn_batch(in_seq, out_seq, is_multi_seq=True)

        if createDyBM:
            return super(BatchLinearDyBM, self)

    def predict(self, in_seq):
        """
        Predict using LinearDyBM without updating its parameters.
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

    def learn_batch(self, in_seq, out_seq=None, is_multi_seq=False):
        """
        Initialize weight matrices of LinearDyBM by batch-learning

        Parameters
        ----------
        in_seq : list of numpy or arraymath array each of shape (in_dim, )
            input sequence
        out_seq : optional, list of numpy or arraymath array each of shape
            (out_dim, ) output sequence

        """

        if out_seq is None:
            out_seq = in_seq
        self._learn_batch(in_seq, out_seq, is_multi_seq)

    def _learn_batch(self, in_seq, out_seq, is_multi_seq=False):
        """
        private method to perform batch learning

        """

        if is_multi_seq is False:
            in_seq = [in_seq]
            out_seq = [out_seq]

        # Handle special case when delay == 1 and len(decay_rates) == 0
        if self.len_fifo <= 0 and self.n_etrace == 0:
            if self._use_bias:
                in_seq_list = list()
                for seq in in_seq:
                    in_seq_list.extend(seq)
                self.variables["b"] = np.average(in_seq_list, axis=0)
            else:
                raise ValueError("delay + len(decay_rates) >= 1 or use_bias must be True")
            return
        # end of Handling special case

        in_seq = [amath.to_numpy(i) for i in in_seq]
        out_seq = [amath.to_numpy(o) for o in out_seq]

        # create feature vector for batch regression
        # L: total length of in_seq
        # N: length used to fit the model
        # seq_number: number of sequences used in learning
        # seq_lens: list of length of each sequence
        # X: array, shape (N, len_fifo*in_dim + n_etrace*in_dim).
        # Y: array, shape (N, out_dim).
        L = 0
        N = 0
        seq_number = len(in_seq)
        seq_lens = list()
        for i in xrange(seq_number):
            seq_lens.append(len(in_seq[i]))
            L += seq_lens[-1]
            N += seq_lens[-1] - 1 if self.learn_beginning is True else seq_lens[-1] - self.len_fifo

        # Initialize array X and Y
        if self.esn is None:
            # in_seq[0] ... in_seq[T-2]
            X = np.zeros((N, self.len_fifo * self.in_dim + self.n_etrace * self.in_dim))
        else:
            # in_seq[0] ... in_seq[T-2]
            X = np.zeros((N, self.len_fifo * self.in_dim +
                          self.n_etrace * self.in_dim + self.esn.out_dim))
        Y = np.zeros((N, self.out_dim))  # out_seq[1] ... out_seq[T-1]

        start_location = 1 if self.learn_beginning is True else self.len_fifo
        # Initialize Y
        index = 0
        for i in xrange(seq_number):
            for t in xrange(start_location, seq_lens[i]):
                Y[index] = out_seq[i][t]
                index += 1

        # Initialize X
        OFFSET = self.in_dim * self.len_fifo
        ESNOFFSET = OFFSET + self.n_etrace * self.in_dim
        # to store the k-th decay vectors or e_trace
        decayVectors = [np.zeros((1, self.in_dim)) for k in range(self.n_etrace)]
        # decayVectors[k] at time t is self.decay_rates[k] * decayVectors[k]
        # at time t - 1

        decay_rates = amath.to_numpy(self.decay_rates)
        index = 0
        for i in xrange(seq_number):
            for t in xrange(start_location, seq_lens[i]):
                seq_index = t - 1
                if self.insert_to_etrace == "w_delay" and (t - self.len_fifo) >= 0:
                    for k in range(len(decayVectors)):
                        decayVectors[k] = decayVectors[k] * \
                            decay_rates[k] + in_seq[i][seq_index - self.len_fifo]
                if self.insert_to_etrace == "wo_delay":
                    # update eligibility trace without delay
                    for k in range(len(decayVectors)):
                        decayVectors[k] = decayVectors[k] * \
                            decay_rates[k] + in_seq[i][seq_index]

                for l in xrange(self.len_fifo):
                    if (seq_index - l) < 0:  # no data available
                        break
                    X[index, self.in_dim * l: self.in_dim * (l + 1)] = in_seq[i][seq_index - l]

                for k in xrange(self.n_etrace):
                    X[index, OFFSET + self.in_dim * k:OFFSET +
                        self.in_dim * (k + 1)] = decayVectors[k]

                if self.esn is not None:
                    self.esn._update_state(in_seq[i][seq_index])
                    X[index, ESNOFFSET:] = self.esn.si

                index += 1

        # by Ridge, Lasso, or MultiTaskLasso, we will obtain
        # W': array, shape(out_dim, len_fifo*in_dim + n_etrace*in_dim) a weight matrix
        # b': array, shape(out_dim, ) intercept
        # satisfying Y^T = W' X^T + b'

        # choose batch method
        if self.batch_method == "Ridge":
            model = sklearn.linear_model.Ridge(alpha=self.L2["V"],
                                               normalize=True, random_state=0,
                                               max_iter=10000,
                                               fit_intercept=self._use_bias)
        elif self.batch_method == "Lasso":
            model = sklearn.linear_model.Lasso(alpha=self.L1["V"],
                                               normalize=True, random_state=0,
                                               max_iter=10000, warm_start=True,
                                               positive=self._positive,
                                               fit_intercept=self._use_bias)
        elif self.batch_method == "MultiTaskLasso":
            model = sklearn.linear_model.MultiTaskLasso(alpha=self.L1["V"],
                                                        normalize=True,
                                                        random_state=0,
                                                        max_iter=10000,
                                                        warm_start=True,
                                                        fit_intercept=self._use_bias)
        else:
            model = sklearn.linear_model.Ridge(alpha=self.L2["V"],
                                               normalize=True, random_state=0,
                                               max_iter=10000,
                                               fit_intercept=self._use_bias)

        # batch learning by fitting X and Y to the model
        model.fit(X, Y)

        # extracting weight matrices
        if self.esn is None:
            weightMatrix = model.coef_.reshape(
                (self.out_dim, self.in_dim * self.len_fifo + self.in_dim * self.n_etrace))
        else:
            weightMatrix = model.coef_.reshape(
                (self.out_dim, self.in_dim * self.len_fifo + self.in_dim * self.n_etrace + self.esn.out_dim))
            self.esn.variables["A"] = amath.array(
                np.transpose(weightMatrix[:, ESNOFFSET:]))
        bVec = model.intercept_

        # partition weight matrices and bVec to DyBM model
        # DyBM's weight matrices W[l] = transpose W'[:,in_dim*l:in_dim*(l+1)]
        # note that transpose is necessary because W[l] array, shape (in_dim,out_dim) for
        # l = 0 ... len_fifo-1
        for l in range(self.len_fifo):
            self.variables["W"][l] = amath.array(np.transpose(
                weightMatrix[:, self.in_dim * l:self.in_dim * (l + 1)]))

        # DyBM's eligibility trace weigth matrices
        # V[k] = W'[:, in_dim*len_fifo+in_dim*k: in_dim*len_fifo+in_dim*(k+1)]

        # V[k] array, shape (in_dim,out_dim) for k = 0 ... n_etrace-1
        for k in range(self.n_etrace):
            self.variables["V"][k] = amath.array(np.transpose(
                weightMatrix[:, OFFSET + self.in_dim * k:OFFSET + self.in_dim * (k + 1)]))

        # DyBM's bias b = b'
        self.variables["b"] = amath.array(bVec)


class BatchMultiTargetLinearDyBM(MultiTargetLinearDyBM):
    """BatchMultiTargetLinearDyBM is a batch version of MultiTargetLinearDyBM.
    Each of its layers is BatchLinearDyBM

    Parameters
    ----------
    in_dim : int
        dimension of input time-series
    out_dims : list,
        list of the dimension of target time-series
    SGDs : list of the object of SGD.SGD, optional
        list of the optimizer for the stochastic gradient method
    delay : int, optional
        length of the FIFO queue plus 1
    decay_rates : list, optional
        decay rates of eligibility traces
    L1 : float, optional
        strength of L1 regularization
    L2 : float, optional
        strength of L2 regularization
    use_bias : boolean, optional
        whether to use bias parameters
    sigma : float, optional
        standard deviation of initial values of weight parameters
    insert_to_etrace : str, {"w_delay", "wo_delay"},  optional
        "w_delay" : insert pattern observed d-1 time steps ago into
        eligibility traces
        "wo_delay" : insert the latest pattern into eligibility traces
    esns : list of ESNs, optional
        Echo state network [NOT SUPPORTED YET]
    batch_methods: list of string for specifying batch_method, optional
        Choose from "Ridge", "Lasso", or "MultiTaskLasso". Default is Ridge
    positives: list of boolean, optional
        If true and batch_method="Lasso", the coefficients are all positive
        values
    """

    def __init__(self, in_dim, out_dims, SGDs=None, delay=2, decay_rates=[0.5],
                 L1=0, L2=0, use_bias=True, sigma=0,
                 insert_to_etrace="wo_delay", esns=None, batch_methods=None,
                 positives=None, learn_beginning=True):

        if SGDs is None:
            SGDs = [AdaGrad() for i in range(len(out_dims))]
        if esns is None:
            esns = [None for i in range(len(out_dims))]
        if batch_methods is None:
            batch_methods = ["Ridge" for i in range(len(out_dims))]
        if positives is None:
            positives = [False for i in range(len(out_dims))]

        if not len(out_dims) == len(SGDs):
            raise ValueError("out_dims and SGDs must have a common length")
        if not len(out_dims) == len(esns):
            raise ValueError("out_dims and esns must have a common length")
        if not len(out_dims) == len(batch_methods):
            raise ValueError("out_dims and batch_methods must have "
                             "a common length")
        if not len(out_dims) == len(positives):
            raise ValueError("out_dims and positives must have a common length")

        self.layers = [BatchLinearDyBM(in_dim, out_dim, delay, decay_rates,
                                       SGD, L1, L2, use_bias, sigma,
                                       insert_to_etrace, esn, batch_method,
                                       positive, learn_beginning)
                       for (out_dim, SGD, esn, batch_method, positive)
                       in zip(out_dims, SGDs, esns, batch_methods, positives)]

        # Only layer 0 has internal states, which are shared among all layers
        for i in xrange(1, len(self.layers)):
            self.layers[i].fifo = self.layers[0].fifo
            self.layers[i].e_trace = self.layers[0].e_trace

        StochasticTimeSeriesModel.__init__(self)

    def fit(self, in_seq, out_seqs, createDyBM=True):
        """
        Fit LinearDyBM with in_seq and out_seqs and set FIFO
        and the eligibility trace

        Parameters
        ----------
        in_seq: list of arraymath array each of shape(in_dim, )
            input sequence
        out_seqs: list of numpy or arraymath array each of shape (out_dim, )
            output sequence
        createDyBM: optional, boolean
            if True, return MultiTargetLinearDyBM fitted to in_seq and out_seq

        Returns
        -------
        MultiTargetLinearDyBM fitted to in_seq and out_seq if createDyBM is True

        """

        if not len(self.layers) == len(out_seqs):
            raise ValueError("out_seqs and layers must have a common length")

        for i in xrange(len(self.layers)):
            self.layers[i].learn_batch(in_seq, out_seqs[i])
        for in_pattern in in_seq:  # only update layers[0] for input pattern
            self.layers[0]._update_state(in_pattern)

        if createDyBM:
            return super(BatchMultiTargetLinearDyBM, self)

    def predict(self, in_seq):
        """
        Predict using LinearDyBM without updating its parameters.
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

        return [layer.predict(in_seq) for layer in self.layers]

    def learn_batch(self, in_seq, out_seqs):
        """
        Initialize weight matrices of DyBM by batch-learning

        Parameters
        ----------
        in_seq : list of numpy or arraymath array each of shape (in_dim, )
            input sequence
        out_seq : list of numpy or arraymath array each of shape (out_dim, )
            output sequence

        """

        for i, layer in enumerate(self.layers):
            layer._learn_batch(in_seq, out_seqs[i])
