# (C) Copyright IBM Corp. 2019
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


"""Implementation of Dynamic Determinantal Point Processes, AAAI-18.

"""

__author__ = "Takayuki Osogami, Rudy Raymond"


import numpy as np
from pydybm.time_series.time_series_model import StochasticTimeSeriesModel
from pydybm.base.sgd import AdaGrad
from numpy.linalg import pinv as npLinalgInv
from copy import deepcopy


class DataDyDPP():

    """
    Data for Dynamic DPP
    """

    def __init__(self, T, D, M, dtype=bool):

        self.T = T           # length of time
        self.D = D           # length of lag
        self.M = M           # number of choices at each time
        self.X = np.zeros((self.M, self.T + self.D), dtype=dtype)
        self.selected = [list() for t in range(self.T)]
        # Notice that X stores the selection in reverse order
        # selection is at t = 0, ..., T-1 (there are T vectors of selections)
        # X[:, 0] stores selection at time T-1, X[:, 1] at time T-2, ...,
        # X[:, T-1] stores at time 0, and the rest is zero
        # X[:, j] stores selection at time T-j-1 for j = 0 ... T-1
        # selection at time t is stored at X[:, T-t-1] for t = 0 ... T-1
        # WARNING: The paper uses t=1...T, while here t=0...T-1.
        # That means X[:, 0] will never be used as X[:,J]

    def set_selection(self, t, aSet):

        """
        Set X[t,i] = 1 if i is an element of aSet

        Parameters:
        ----------
        t: int
            time
        aSet: a set selected at time t
        """

        self.X[:, self.T-1-t] = np.zeros(self.M)  # clear selection
        for s in aSet:
            self.X[s, self.T-1-t] = 1
        self.selected[t] = sorted(aSet)

    def get_selection(self, t):

        """
        Get selection at time t

        Parameters:
        ----------
        t: int
            time

        Return:
        ----------
        arrIndex: numpy.array
            a sorted index of selected items
        """

        if t >= self.T or t < 0:
            return set()

        return self.selected[t]

    def get_max_subset_size(self):

        """
        Get the maximum size of subsets selected from t=0 to t=T-1
        """

        size = list()
        for t in range(self.T):
            selected = self.get_selection(t)
            size.append(len(selected))
        return max(size)


class DyDPP(StochasticTimeSeriesModel):

    """
    Dynamic determinantal point process

    Parameters
    ----------
    dim : int
        the dimension of the input time-series (the size of the ground set)
    rank : int
        the rank of the kernel matrix
    lag : int
        the maximum time lag of dependency
    L1 : float
        not used
    L2 : float
        strength of L2 regularization on W
    """

    def __init__(self, dim, rank, lag, L1=0.1, L2=0.1, data=None,
                 SGD=AdaGrad(), initFactor=0.1):

        self.M = dim
        self.K = rank
        self.D = lag
        self.n_var = self.M * self.K + self.D * self.K
        self.data = None
        self.t = 0
        if data is not None:
            self.data = data
            self.T = data.T
            assert self.data.D == self.D, "lag of data not the same with DyDPP"
            assert self.data.M == self.M, "number of choices not the same"

        self.variables = dict()

        # INITIALIZATION WITH RANDOM VECTORS
        np.random.seed(0)
        self.variables["B"] = np.random.uniform(low=-0.1, high=0.1,
                                                size=(self.M, self.K))
        self.variables["W"] = np.zeros((self.K, self.D))

        # This is a vector for regulazing rows of V(t) and B
        self.regularizer = np.ones(self.M, dtype=float)

        self.init_state()

        self.SGD = SGD.set_shape(self.variables)
        self.L2 = dict()
        self.L2["B"] = 0
        self.L2["W"] = L2

        self.maxGamma = 1e-3
        self.minGamma = 1e-9

    def initialize_W(self, isOne=True, factor=0.1, initW=None):

        oldW = np.copy(self.variables["W"])
        if initW is not None:
            self.variables["W"] = initW
        elif isOne:
            self.variables["W"] = factor * np.ones((self.K, self.D),
                                                   dtype=float)
        else:
            self.variables["W"] = np.zeros((self.K, self.D), dtype=float)

        return oldW

    def set_B(self, B):
        self.variables["B"] = deepcopy(B)

    def initialize_B_backtrack(self, trainset, isResetB=False, minIteration=5,
                               maxIteration=10, eps=-1.0e-2, SGD=AdaGrad(),
                               alpha=0.1, isRegulazingB=False):
        """
        Parameters
        ----------
        trainset: list of DataDyDPP
            list of time-series selection vectors for training
        validset: list of DataDyDPP
            list of time-series selection vectors for validation
        isResetB: if True, self.variables["B"] is reset with samples from
        uniform distribution on [-0.1,0.1] before running initialization
        """

        print("Initializing B with backtrack")
        oldW = self.initialize_W(isOne=False)  # set W=0 and store oldW

        variables = dict()
        if isResetB:
            self.variables["B"] = np.random.uniform(low=-0.1, high=0.1,
                                                    size=(self.M, self.K))
        variables["B"] = np.copy(self.variables["B"])

        if trainset is None:
            trainset = [self.data]

        if isRegulazingB:
            # Compute regularization of B according to Gartrell et al. 2017
            itemFreq = dict()
            for i in range(self.M):
                itemFreq[i] = 1.0  # to avoid division by zero
            for data in trainset:
                for t in range(data.T):
                    selectionT = data.get_selection(t)
                    for i in selectionT:
                        itemFreq[i] += 1.0
            # Set the value of self.regularizer during training of B
            self.regularizer = np.array([1.0/(1.0+itemFreq[i])
                                         for i in range(self.M)])
            del itemFreq

        bestLL = self.get_average_LL(trainset)
        print("bestLL", bestLL)

        nIters = 0
        for nIters in range(maxIteration):
            delta = dict()
            delta["B"] = np.zeros(variables["B"].shape)
            for data in trainset:
                delta["B"] = delta["B"] - data.T * variables["B"].dot(np.linalg.inv(np.eye(self.K) + variables["B"].T.dot(variables["B"])))
                for t in range(data.T):
                    selectionT = data.get_selection(t)
                    if len(selectionT) <= 0:
                        continue
                    delta["B"][selectionT, :] = delta["B"][selectionT, :] + npLinalgInv(variables["B"][selectionT, :]).T

            if isRegulazingB:
                delta["B"] = delta["B"] - alpha * np.diag(self.regularizer).dot(variables["B"])

            gamma = self.maxGamma
            while gamma > self.minGamma:
                self.variables["B"] = variables["B"] + gamma * delta["B"]
                LL = self.get_average_LL(trainset)
                print("nIters:", nIters,
                      "gamma:", gamma,
                      "LL:", LL,
                      "bestLL:", bestLL)
                if LL > bestLL:
                    print("breaking while with", LL, bestLL)
                    variables["B"] = deepcopy(self.variables["B"])
                    bestLL = LL

                    if gamma == self.maxGamma:
                        self.maxGamma *= 10
                    if gamma < self.maxGamma / 10.:
                        self.maxGamma = gamma * 10
                    break
                gamma = gamma / 10.

            print(nIters, bestLL)
            if gamma <= self.minGamma:
                print("breaking for with", gamma, self.minGamma)
                break

        self.variables["B"] = variables["B"]

        self.initialize_W(initW=oldW)  # restore oldW

        return bestLL

    def learn_dataset_backtrack(self, trainset, isResetB=False, minIteration=5,
                                maxIteration=10, eps=-1.0e-2, SGD=AdaGrad(),
                                alpha=0.1, isRegulazingB=False):

        variables = deepcopy(self.variables)

        bestLL = self.get_average_LL(trainset)
        print("bestLL", bestLL)

        nIters = 0
        for nIters in range(maxIteration):
            delta = dict()
            for data in trainset:
                self.set_data(data)
                for t in range(data.T):
                    gradient = self.get_gradient(t, applyL2=False, alpha=alpha,
                                                 isRegulazingB=False)
                    for key in gradient:
                        if key in delta:
                            delta[key] = delta[key] + gradient[key]
                        else:
                            delta[key] = gradient[key]

            if isRegulazingB:
                delta["B"] = delta["B"] - alpha * np.diag(self.regularizer).dot(variables["B"])
            gradient["W"] = gradient["W"] - self.L2["W"] * self.variables["W"]

            gamma = self.maxGamma
            while gamma > self.minGamma:
                self.variables["B"] = variables["B"] + gamma * delta["B"]
                self.variables["W"] = variables["W"] + gamma * delta["W"]
                LL = self.get_average_LL(trainset)
                print("nIters:", nIters,
                      "gamma:", gamma,
                      "LL:", LL,
                      "bestLL:", bestLL)
                if LL > bestLL:
                    print("breaking while with", LL, bestLL)
                    variables = deepcopy(self.variables)
                    bestLL = LL

                    if gamma == self.maxGamma:
                        self.maxGamma *= 10
                    break
                gamma = gamma / 10.

            print(nIters, bestLL)
            if gamma <= self.minGamma:
                print("breaking for with", gamma, self.minGamma)
                break

        return bestLL

    def __get_V(self, t=None):
        if t is None:
            t = self.T
        J = np.arange(self.T - t, self.T - t + self.D)
        V = self.variables["B"] + self.data.X[:, J].dot(self.variables["W"].T)
        return V, J

    def __slice_matrix(self, A, rows, cols):
        if len(rows) <= 0 or len(cols) <= 0:
            return None
        else:
            answer = A[rows, :]
            return answer[:, cols]

    def __slice_VVT(self, V, rows, cols):
        if len(rows) <= 0 or len(cols) <= 0:
            return None
        else:
            answer = V[rows, :].dot(V[cols, :].T)
            return answer

    def get_gradient(self, t=None, applyL2=False, alpha=1.0,
                     isRegulazingB=False):

        """
        Parameters
        ----------
        t : int
           subset selected at time t will be used to update the parameters
        """

        if t is None:
            t = self.t
        # Prepare basic matrices
        V, J = self.__get_V(t)
        invVTVplusI = np.linalg.inv(V.T.dot(V) + np.eye(V.shape[1]))
        R = V.dot(invVTVplusI)
        # Gradient
        gradient = dict()
        gradient["B"] = -1.0*R
        tSelection = self.data.get_selection(t)
        if len(tSelection) <= 0:  # in case no selection is made, no gradient
            gradient["W"] = np.zeros(self.variables["W"].shape)
        else:
            ATilde = npLinalgInv(V[tSelection, :])
            gradient["B"][tSelection, : ] = gradient["B"][tSelection, : ] + ATilde.T
            gradient["W"] = ATilde.dot(self.__slice_matrix(self.data.X, tSelection, J)) - R.T.dot(self.data.X[:, J])

        if isRegulazingB:
            gradient["B"] = gradient["B"] - alpha * np.diag(self.regularizer).dot(V)
            gradient["W"] = gradient["W"] - alpha * np.diag(self.regularizer).dot(V).T.dot(self.data.X[:, J])

        if applyL2:
            # We want to apply L2 regularization only to W
            # L2 regularization on B may be more likely to result in det=0
            # particularly when self.L2 is too large
            gradient["W"] = gradient["W"] - self.L2["W"] * self.variables["W"]

        return gradient

    def _update_parameters(self, delta):  # delta computed from Nesterov
        for key in delta:
            self.variables[key] = self.variables[key] + delta[key]

    def get_LL(self, data=None):
        if data is None:
            data = self.data
        lLL = []
        for t in range(0, self.T):
            V, J = self.__get_V(t)
            tSelection = np.array(data.get_selection(t))
            LtXt = self.__slice_VVT(V, tSelection, tSelection)
            Ct = V.T.dot(V)
            if LtXt is None:
                s1, v1 = 0, 0
            else:
                s1, v1 = np.linalg.slogdet(LtXt)
            if s1 == 0 and np.isinf(v1):
                v1 = v1
                print("LOG DETERMINANT IS ZERO: " + str(LtXt) + str(s1) + str(v1))
            s2, v2 = np.linalg.slogdet(np.eye(Ct.shape[0]) + Ct)
            lLL.append(v1 - v2)
        return np.sum(lLL)/len(lLL)

    def get_average_LL(self, dataset):
        """
        compute average loglikelihood

        dataset : list of DataDyDPP
        """
        old_data = deepcopy(self.data)  # store current data

        llList = []
        nDataList = []
        for i, data in enumerate(dataset):
            self.set_data(data)
            logLikelihood = self.get_LL()
            llList.append(logLikelihood)
            nDataList.append(data.T)
        aveLL = np.sum(np.array(llList) * np.array(nDataList)) / np.sum(nDataList)

        if old_data is None:
            self.data = None
        else:
            self.set_data(old_data)  # restore data

        return aveLL

    def _get_sample(self):
        pass

    def init_state(self):
        self.t = 0

    def _update_state(self):
        self.t = self.t + 1

    def set_data(self, data):
        self.data = data
        self.T = data.T
        assert self.data.D == self.D, "lag of data not the same with DyPP"
        assert self.data.M == self.M, "number of choices not the same"

    def set_learning_rate(self, learningRate):
        self.SGD.set_learning_rate(learningRate)
