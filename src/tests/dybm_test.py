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

""" DyBM_test """

__author__ = "Takayuki Osogami"


import unittest
from six.moves import xrange
import numpy as np
import tests.simple
from tests.arraymath import NumpyTestMixin, CupyTestMixin
import pydybm.arraymath as amath
from pydybm.time_series.dybm import LinearDyBM, BinaryDyBM
from pydybm.base.sgd import AdaGrad, ADAM
from pydybm.base.generator import Uniform


class DyBMTestCase(object):
    """ unit test for VectorLogisticRegression
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
        self.max_repeat = 100000
        self.in_dim = 3     # dimension of input sequence
        self.out_dim = 2    # dimension of target sequence
        self.rate = 0.01    # learning rate

    def tearDown(self):
        pass

    def test_GenerativeLinearDyBM(self):
        """ testing minimal consistency in learning a sequence
        """
        print("\nDyBMTestCase.testGenerativeGaussianDyBM")
        for delay in [1, 3]:
            for SGD in [AdaGrad, ADAM]:
                for GENERATOR in [True, False]:
                    model = LinearDyBM(self.in_dim, self.in_dim, delay,
                                       SGD=SGD())
                    model.set_learning_rate(self.rate)
                    i = tests.simple.test_real_model(model, self.max_repeat,
                                                     GENERATOR)
                    self.assertLess(i, self.max_repeat)

    def test_DiscriminativeLinearDyBM(self):
        """ testing minimal consistency in learning a sequence to an output
        """
        print("\nDyBMTestCase.testDiscriminativeGaussianDyBM")

        for delay in [1, 3]:
            for SGD in [AdaGrad, ADAM]:
                for GENERATOR in [True, False]:
                    model = LinearDyBM(self.in_dim, self.out_dim, delay,
                                       SGD=SGD())
                    model.set_learning_rate(self.rate)
                    i = tests.simple.test_real_model(model, self.max_repeat,
                                                     GENERATOR)
                    self.assertLess(i, self.max_repeat)

    def test_GenerativeBinaryDyBM(self):
        """ testing minimal consistency in learning a sequence
        """
        print("\nDyBMTestCase.testGenerativeBinaryDyBM")
        for delay in [1, 3]:
            for SGD in [AdaGrad]:
                for GENERATOR in [True]:
                    model = BinaryDyBM(self.in_dim, self.in_dim, delay,
                                       SGD=SGD())
                    model.set_learning_rate(1.)
                    i = tests.simple.test_binary_model(model, self.max_repeat,
                                                       GENERATOR)
                    self.assertLess(i, self.max_repeat)

    def test_DiscriminativeBinaryDyBM(self):
        """ testing minimal consistency in learning a sequence to an output
        """
        print("\nDyBMTestCase.testDiscriminativeBinaryDyBM")
        for delay in [1, 3]:
            for SGD in [AdaGrad]:
                for GENERATOR in [True]:
                    model = BinaryDyBM(self.in_dim, self.out_dim, delay,
                                       SGD=SGD())
                    model.set_learning_rate(1.)
                    i = tests.simple.test_binary_model(model, self.max_repeat,
                                                       GENERATOR)
                    self.assertLess(i, self.max_repeat)

    def test_LearnGenerator(self):
        """ testing learning with generator
        """
        print("\nDyBMTestCase.testLearnGenerator")
        batch = 3
        in_mean = 1.0
        out_mean = 2.0
        d = 0.01
        delay = 1
        rates = [0.5, 0.8]
        L1 = 0.0
        L2 = 0.0

        random = amath.random.RandomState(0)
        in_gen = Uniform(length=batch, low=in_mean - d, high=in_mean + d,
                         dim=self.in_dim)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))

        model = LinearDyBM(self.in_dim, self.in_dim, delay=delay,
                           decay_rates=rates, L1=L1, L2=L2)
        model.set_learning_rate(0.1)
        model._learn_sequence(in_seq)

        model2 = LinearDyBM(self.in_dim, self.in_dim, delay=delay,
                            decay_rates=rates, L1=L1, L2=L2)
        model2.set_learning_rate(0.1)
        model2.learn(in_gen)

        random = amath.random.RandomState(0)
        in_gen = Uniform(length=batch, low=in_mean - d, high=in_mean + d,
                         dim=self.in_dim)
        in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                                size=(batch, self.in_dim))

        random = amath.random.RandomState(0)
        out_gen = Uniform(length=batch, low=out_mean - d, high=out_mean + d,
                          dim=self.out_dim)
        out_seq = random.uniform(low=out_mean - d, high=out_mean + d,
                                 size=(batch, self.out_dim))

        self.assertEqual(model.variables.keys(), model2.variables.keys())
        for key in model.variables:
            self.assertTrue((model.variables[key] ==
                             model2.variables[key]).all())

        model = LinearDyBM(self.in_dim, self.out_dim, delay=delay,
                           decay_rates=rates, L1=L1, L2=L2)
        model.set_learning_rate(0.1)
        model._learn_sequence(in_seq, out_seq)

        model2 = LinearDyBM(self.in_dim, self.out_dim, delay=delay,
                            decay_rates=rates, L1=L1, L2=L2)
        model2.set_learning_rate(0.1)
        model2.learn(in_gen, out_gen)

        self.assertEqual(model.variables.keys(), model2.variables.keys())
        for key in model.variables:
            self.assertTrue((model.variables[key] ==
                             model2.variables[key]).all())

    def test_UpdateState(self):
        """ testing fifo, eligibility trace, and update_state method in
        LinearDyBM
        """
        print("\n * testing fifo, eligibility trace, and update_state method"
              " in LinearDyBM \n")
        in_dim = 3
        delay = 3
        decay_rate = 0.5

        len_ts = 10

        print("testing wo_delay, single e_trace")
        model = LinearDyBM(in_dim, delay=delay, decay_rates=[decay_rate],
                           insert_to_etrace="wo_delay")
        random = np.random.RandomState(0)
        in_patterns = np.random.uniform(size=(len_ts, in_dim))
        fifo_test = np.zeros((delay - 1, in_dim))
        e_trace_test = np.zeros((1, in_dim))
        for i in xrange(len_ts):
            self.assertTrue(np.allclose(amath.to_numpy(model.fifo.to_array()),
                                        fifo_test))
            self.assertTrue(np.allclose(amath.to_numpy(model.e_trace),
                                        e_trace_test))
            model.learn_one_step(amath.array(in_patterns[i]))
            model._update_state(amath.array(in_patterns[i]))

            fifo_test[1:] = fifo_test[:-1]
            fifo_test[0] = in_patterns[i]
            e_trace_test = e_trace_test * decay_rate + in_patterns[i]

        print("testing w_delay, single e_trace")
        model = LinearDyBM(in_dim, delay=delay, decay_rates=[decay_rate],
                           insert_to_etrace="w_delay")
        random = np.random.RandomState(0)
        in_patterns = np.random.uniform(size=(len_ts, in_dim))
        fifo_test = np.zeros((delay - 1, in_dim))
        e_trace_test = np.zeros((1, in_dim))
        for i in xrange(len_ts):
            self.assertTrue(np.allclose(amath.to_numpy(model.fifo.to_array()),
                                        fifo_test))
            self.assertTrue(np.allclose(amath.to_numpy(model.e_trace),
                                        e_trace_test))
            model.learn_one_step(amath.array(in_patterns[i]))
            model._update_state(amath.array(in_patterns[i]))

            fifo_test[1:] = fifo_test[:-1]
            fifo_test[0] = in_patterns[i]
            if i < delay - 1:
                pass
            else:
                e_trace_test = e_trace_test * decay_rate \
                    + in_patterns[i - delay + 1]

        print("testing w_delay, two e_traces")
        model = LinearDyBM(in_dim, delay=delay,
                           decay_rates=[decay_rate, decay_rate**2],
                           insert_to_etrace="w_delay")
        random = np.random.RandomState(0)
        in_patterns = np.random.uniform(size=(len_ts, in_dim))
        fifo_test = np.zeros((delay - 1, in_dim))
        e_trace_test = np.zeros((2, in_dim))
        for i in xrange(len_ts):
            self.assertTrue(np.allclose(amath.to_numpy(model.fifo.to_array()),
                                        fifo_test))
            self.assertTrue(np.allclose(amath.to_numpy(model.e_trace),
                                        e_trace_test))
            model.learn_one_step(amath.array(in_patterns[i]))
            model._update_state(amath.array(in_patterns[i]))

            fifo_test[1:] = fifo_test[:-1]
            fifo_test[0] = in_patterns[i]
            if i < delay - 1:
                pass
            else:
                e_trace_test[0] = e_trace_test[0] * decay_rate \
                    + in_patterns[i - delay + 1]
                e_trace_test[1] = e_trace_test[1] * (decay_rate**2) \
                    + in_patterns[i - delay + 1]

        print("testing w_delay, single e_trace, delay=1")
        delay = 1
        model = LinearDyBM(in_dim, delay=delay, decay_rates=[decay_rate],
                           insert_to_etrace="w_delay")
        random = np.random.RandomState(0)
        in_patterns = random.uniform(size=(len_ts, in_dim))
        e_trace_test = np.zeros((1, in_dim))
        for i in xrange(len_ts):
            self.assertTrue(np.allclose(amath.to_numpy(model.e_trace),
                                        e_trace_test))
            model.learn_one_step(amath.array(in_patterns[i]))
            model._update_state(amath.array(in_patterns[i]))

            if i < delay - 1:
                pass
            else:
                e_trace_test = e_trace_test * decay_rate \
                    + in_patterns[i - delay + 1]


class DyBMTestCaseNumpy(NumpyTestMixin, DyBMTestCase, unittest.TestCase):
    pass


class DyBMTestCaseCupy(CupyTestMixin, DyBMTestCase, unittest.TestCase):
    pass


if __name__ == "__main__":

    unittest.main()

