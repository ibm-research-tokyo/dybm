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

__author__ = "Rudy Raymond, Kun Zhao"


import unittest
import tests.simple
from tests.arraymath import NumpyTestMixin, CupyTestMixin
from six.moves import xrange
import pydybm.arraymath as amath
from pydybm.time_series.dybm import LinearDyBM
from pydybm.time_series.batch_dybm import BatchLinearDyBM
from pydybm.base.sgd import AdaGrad
from pydybm.base.generator import Uniform
from pydybm.base.generator import NoisySin
from pydybm.base.metrics import RMSE


class BatchDyBMTestCase(object):
    """ unit test for BatchDyBM
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
        self.max_repeat = 10000
        self.in_dim = 3     # dimension of input sequence
        self.out_dim = 2    # dimension of target sequence
        self.rate = 0.01    # learning rate

    def tearDown(self):
        pass

    def test_GenerativeBatchLinearDyBM(self):
        """ testing minimal consistency in learning a sequence
        """
        print("\nDyBMTestCase.testGenerativeBatchLinearDyBM")
        for delay in [1, 3]:
            for SGD in [AdaGrad]:
                for GENERATOR in [True]:
                    model = BatchLinearDyBM(self.in_dim, self.in_dim, delay,
                                            SGD=SGD())
                    model.set_learning_rate(self.rate)
                    i = tests.simple.test_real_model(model, self.max_repeat,
                                                     GENERATOR)
                    self.assertLess(i, self.max_repeat)

    def test_BatchDiscriminativeLinearDyBM(self):
        """ testing minimal consistency in learning a sequence to an output
        """
        print("\nDyBMTestCase.testBatchDiscriminativeLinearDyBM")

        for delay in [1, 3]:
            for SGD in [AdaGrad]:
                for GENERATOR in [True]:
                    model = BatchLinearDyBM(self.in_dim, self.out_dim, delay,
                                            SGD=SGD())
                    model.set_learning_rate(self.rate)
                    i = tests.simple.test_real_model(model, self.max_repeat,
                                                     GENERATOR)
                    self.assertLess(i, self.max_repeat)

    def _compute_rmse(self, prediction, target):
        a = amath.array(prediction.to_list(), dtype=float)
        b = amath.array(target)
        rmse = amath.sqrt(amath.mean((a-b)**2))
        return rmse

    def test_LearnSequenceBatch(self):
        """ testing learning with Sequence
        """
        print("\nBatchDyBMTestCase.testLearnSequenceBatch")
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
        model = BatchLinearDyBM(self.in_dim, self.in_dim, delay=delay,
                                decay_rates=rates, L1=L1, L2=L2,
                                insert_to_etrace="wo_delay")
        model.set_learning_rate(0.1)
        model.learn_batch(in_seq)
        model.fit(in_seq)
        model.init_state()
        prediction = model.predict(in_seq)
        print("RMSE:", self._compute_rmse(prediction, in_seq))


        random = amath.random.RandomState(0)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))
        random = amath.random.RandomState(0)
        out_seq = random.uniform(low=out_mean - d, high=out_mean + d,
                                 size=(batch, self.out_dim))

        model = BatchLinearDyBM(self.in_dim, self.out_dim, delay=delay,
                                decay_rates=rates, L1=L1, L2=L2,
                                insert_to_etrace="wo_delay")
        model.set_learning_rate(0.1)
        model.learn_batch(in_seq, out_seq)
        model.fit(in_seq, out_seq)
        model.init_state()
        prediction = model.predict(in_seq)
        print("RMSE:", self._compute_rmse(prediction, out_seq))

        print("\n * testing w_delay")
        random = amath.random.RandomState(0)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))
        model = BatchLinearDyBM(self.in_dim, self.in_dim, delay=delay,
                                decay_rates=rates, L1=L1, L2=L2,
                                insert_to_etrace="w_delay")
        model.set_learning_rate(0.1)
        model.learn_batch(in_seq)
        model.fit(in_seq)
        model.init_state()
        prediction = model.predict(in_seq)
        print("RMSE:", self._compute_rmse(prediction, in_seq))

        random = amath.random.RandomState(0)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))
        random = amath.random.RandomState(0)
        out_seq = random.uniform(low=out_mean - d, high=out_mean + d,
                                 size=(batch, self.out_dim))

        model = BatchLinearDyBM(self.in_dim, self.out_dim, delay=delay,
                                decay_rates=rates, L1=L1, L2=L2,
                                insert_to_etrace="w_delay")
        model.set_learning_rate(0.1)
        model.learn_batch(in_seq, out_seq)
        model.fit(in_seq, out_seq)
        model.init_state()
        prediction = model.predict(in_seq)
        print("RMSE:", self._compute_rmse(prediction, out_seq))

        print("\n * testing w_delay using Lasso")
        random = amath.random.RandomState(0)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))
        model = BatchLinearDyBM(self.in_dim, self.in_dim, delay=delay,
                                decay_rates=rates, L1=L1, L2=L2,
                                insert_to_etrace="w_delay", batch_method="Lasso")
        model.set_learning_rate(0.1)
        model.learn_batch(in_seq)
        model.fit(in_seq)
        model.init_state()
        prediction = model.predict(in_seq)
        print("RMSE:", self._compute_rmse(prediction, in_seq))

        random = amath.random.RandomState(0)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))
        random = amath.random.RandomState(0)
        out_seq = random.uniform(low=out_mean - d, high=out_mean + d,
                                 size=(batch, self.out_dim))

        model = BatchLinearDyBM(self.in_dim, self.out_dim, delay=delay,
                                decay_rates=rates, L1=L1, L2=L2,
                                insert_to_etrace="w_delay", batch_method="Lasso")
        model.set_learning_rate(0.1)
        model.learn_batch(in_seq, out_seq)
        model.fit(in_seq, out_seq)
        model.init_state()
        prediction = model.predict(in_seq)
        print("RMSE:", self._compute_rmse(prediction, out_seq))


    def test_LearnBeginningMode(self):
        """ testing learn_beginning mode.
        """
        print("\nDyBMTestCase.testLearnFirstMode")

        length = 300
        period = 60
        std = 0
        dim = 1
        data = NoisySin(length,period,std,dim).to_list()

        print("\nlearn_beginning = True (Default mode)")
        for d in xrange(10,20):
            model = BatchLinearDyBM(dim, delay=d, learn_beginning=True)
            model.fit(data)
            result = model.get_predictions(data)
            rmse = RMSE(result[d:], data[d:])
            print("Training RMSE", rmse)
            self.assertLessEqual(rmse, 0.1)

        print("\nlearn_beginning = False")
        for d in xrange(10,20):
            model = BatchLinearDyBM(dim, delay=d, learn_beginning=False)
            model.fit(data)
            result = model.get_predictions(data)
            rmse = RMSE(result[d:], data[d:])
            print("Training RMSE", rmse)
            self.assertLessEqual(rmse, 0.1)

    def test_LearnMultiSequences(self):
        """ testing learn_beginning mode.
        """        
        print("\nDyBMTestCase.testLearnMultiSequences")

        seqs = list()
        for i in xrange(10):
            length = 300
            period = 60
            std = 0.1 * i
            dim = 1
            data = NoisySin(length,period,std,dim).to_list()
            seqs.append(data)

        for d in xrange(10,20):
            model = BatchLinearDyBM(dim, delay=d, learn_beginning=True)
            model.fit_multi_seqs(seqs)
            result = model.get_predictions(seqs[0])
            rmse = RMSE(result[d:], seqs[0][d:])
            print("Training RMSE", rmse)
            self.assertLessEqual(rmse, 0.2)


class BatchDyBMTestCaseNumpy(NumpyTestMixin,
                             BatchDyBMTestCase,
                             unittest.TestCase):
    pass


class BatchDyBMTestCaseCupy(CupyTestMixin,
                            BatchDyBMTestCase,
                            unittest.TestCase):
    pass


if __name__ == "__main__":

    unittest.main()
