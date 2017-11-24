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

""" BatchDyBM_test """

__author__ = "Rudy Raymond"


import unittest
import tests.simple
from tests.arraymath import NumpyTestMixin, CupyTestMixin
import pydybm.arraymath as amath
from pydybm.time_series.dybm import LinearDyBM
from pydybm.time_series.batch_dybm import BatchMultiTargetLinearDyBM
from pydybm.base.sgd import AdaGrad
from pydybm.base.generator import Uniform


class BatchMultiDyBMTestCase(object):
    """ unit test for BatchMultiTargetLinearDyBM
    """

    """
    Attributes
    ----------
    max_repeat : int
        maximum number of training iterations
    in_dim : int
        dimension of input sequence
    out_dim : int
        dimension of target sequence
    """

    def setUp(self):
        self.max_repeat = 20000
        self.in_dim = 3     # dimension of input sequence
        self.out_dim = 2    # dimension of target sequence
        self.rate = 0.1    # learning rate
        self.rates = [self.rate, self.rate]

    def tearDown(self):
        pass


    def _compute_rmse(self, predictions, targets):
        rmse = 0.0
        for i, pred in enumerate(predictions):
            #print(type(amath.array(pred.to_list())), amath.array(pred.to_list()))
            #print(type(amath.array(targets[i])), amath.array(targets[i]))
            a = amath.array(pred.to_list(), dtype=float)
            b = amath.array(targets[i])
            rmse += amath.sqrt(amath.mean((a - b)**2))
        return rmse

    def test_LearnSequenceBatch(self):
        """ testing learning with Sequence
        """
        print("\nBatchMultiDyBMTestCase.testLearnSequenceBatch")
        batch = 3
        in_mean = 1.0
        out_mean = 2.0
        d = 0.01
        delay = 1
        rates = [0.5, 0.8]
        L1 = 0.1
        L2 = 0.1

        print("\n * testing wo_delay")
        random = amath.random.RandomState(0)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))
        model = BatchMultiTargetLinearDyBM(self.in_dim, [self.in_dim, self.in_dim], delay=delay,
                                           decay_rates=rates, L1=L1, L2=L2,
                                           insert_to_etrace="wo_delay")
        model.set_learning_rate([0.1, 0.2])
        model.learn_batch(in_seq, [in_seq, in_seq])
        model.fit(in_seq, [in_seq, in_seq])
        model.init_state()
        predictions = model.predict(in_seq)
        print("RMSE:", self._compute_rmse(predictions, [in_seq, in_seq]))

        random = amath.random.RandomState(0)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))
        random = amath.random.RandomState(0)
        out_seq = random.uniform(low=out_mean - d, high=out_mean + d,
                                 size=(batch, self.out_dim))

        model = BatchMultiTargetLinearDyBM(self.in_dim, [self.out_dim, self.out_dim], delay=delay,
                                           decay_rates=rates, L1=L1, L2=L2,
                                           insert_to_etrace="wo_delay")
        model.set_learning_rate([0.1, 0.2])
        model.learn_batch(in_seq, [out_seq, out_seq])
        model.fit(in_seq, [out_seq, out_seq])
        model.init_state()
        predictions = model.predict(in_seq)
        print("RMSE:", self._compute_rmse(predictions, [out_seq, out_seq]))

        print("\n * testing w_delay")
        random = amath.random.RandomState(0)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))
        model = BatchMultiTargetLinearDyBM(self.in_dim, [self.in_dim, self.in_dim], delay=delay,
                                           decay_rates=rates, L1=L1, L2=L2,
                                           insert_to_etrace="w_delay")
        model.set_learning_rate([0.1, 0.2])
        model.learn_batch(in_seq, [in_seq, in_seq])
        model.fit(in_seq, [in_seq, in_seq])
        model.init_state()
        predictions = model.predict(in_seq)
        print("RMSE:", self._compute_rmse(predictions, [in_seq, in_seq]))

        random = amath.random.RandomState(0)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))
        random = amath.random.RandomState(0)
        out_seq = random.uniform(low=out_mean - d, high=out_mean + d,
                                 size=(batch, self.out_dim))

        model = BatchMultiTargetLinearDyBM(self.in_dim, [self.out_dim, self.out_dim], delay=delay,
                                           decay_rates=rates, L1=L1, L2=L2,
                                           insert_to_etrace="w_delay")
        model.set_learning_rate([0.1, 0.2])
        model.learn_batch(in_seq, [out_seq, out_seq])
        model.fit(in_seq, [out_seq, out_seq])
        model.init_state()
        predictions = model.predict(in_seq)
        print("RMSE:", self._compute_rmse(predictions, [out_seq, out_seq]))

        print("\n * testing w_delay using Lasso")
        random = amath.random.RandomState(0)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))
        model = BatchMultiTargetLinearDyBM(self.in_dim, [self.in_dim, self.in_dim], delay=delay,
                                           decay_rates=rates, L1=L1, L2=L2,
                                           insert_to_etrace="w_delay", batch_methods=["Lasso", "Ridge"])
        model.set_learning_rate([0.1, 0.2])
        model.learn_batch(in_seq, [in_seq, in_seq])
        model.fit(in_seq, [in_seq, in_seq])
        model.init_state()
        predictions = model.predict(in_seq)
        print("RMSE:", self._compute_rmse(predictions, [in_seq, in_seq]))

        random = amath.random.RandomState(0)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))
        random = amath.random.RandomState(0)
        out_seq = random.uniform(low=out_mean - d, high=out_mean + d,
                                 size=(batch, self.out_dim))

        model = BatchMultiTargetLinearDyBM(self.in_dim, [self.out_dim, self.out_dim], delay=delay,
                                           decay_rates=rates, L1=L1, L2=L2,
                                           insert_to_etrace="w_delay", batch_methods=["Lasso", "MultiTaskLasso"])
        model.set_learning_rate([0.1, 0.2])
        model.learn_batch(in_seq, [out_seq, out_seq])
        model.init_state()
        model.fit(in_seq, [out_seq, out_seq])
        predictions = model.predict(in_seq)
        print("RMSE:", self._compute_rmse(predictions, [out_seq, out_seq]))


class BatchDyBMTestCaseNumpy(NumpyTestMixin,
                             BatchMultiDyBMTestCase,
                             unittest.TestCase):
    pass


class BatchDyBMTestCaseCupy(CupyTestMixin,
                            BatchMultiDyBMTestCase,
                            unittest.TestCase):
    pass


if __name__ == "__main__":

    unittest.main()
