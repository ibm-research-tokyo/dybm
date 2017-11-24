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


__author__ = "Takayuki Osogami"


import unittest
import numpy as np
from six.moves import xrange

import tests.simple
from tests.arraymath import NumpyTestMixin, CupyTestMixin
import pydybm.arraymath as amath
from pydybm.time_series.vector_regression import VectorRegressionWithVariance
from pydybm.time_series.vector_regression import VectorLogisticRegression
from pydybm.time_series.vector_regression import MultiTargetVectorRegression
from pydybm.base.sgd import AdaGrad


class VectorRegressionTestCase(object):
    """
    unit test for VectorRegression
    """

    def setUp(self):
        self.max_repeat = 100000
        self.in_dim = 3     # dimension of input sequence
        self.out_dim = 2    # dimension of target sequence
        self.rate = 0.1    # learning rate

    def tearDown(self):
        pass

    def testGenerative(self):
        """
        testing minimal consistency in learning a sequence
        """
        print("VectorRegressionTestCase.testGenerative")
        for order in [0, 2]:
            for SGD in [AdaGrad]:
                model = VectorRegressionWithVariance(
                    self.in_dim, self.in_dim, order, SGD())
                model.set_learning_rate(self.rate)
                i = tests.simple.test_real_model(model, self.max_repeat,
                                                 True)
                self.assertLess(i, self.max_repeat)

    def testDiscriminative(self):
        """
        testing minimal consistency in learning a sequence to an output
        """
        print("VectorRegressionTestCase.testDiscriminative")
        for order in [0, 2]:
            for SGD in [AdaGrad]:
                model = VectorRegressionWithVariance(self.in_dim, self.out_dim, order,
                                                     SGD())
                model.set_learning_rate(self.rate)
                i = tests.simple.test_real_model(model, self.max_repeat,
                                                 True)
                self.assertLess(i, self.max_repeat)

    def testFifo(self):
        """
        testing fifo and _update_state method in VectorRegressionWithVariance
        """
        print("\n * testing fifo and update_state method "
              "in VectorRegressionWithVariance \n")
        in_dim = 3
        order = 3

        len_ts = 10

        model = VectorRegressionWithVariance(in_dim, in_dim, order)
        random = amath.random.RandomState(0)
        in_patterns = amath.random.uniform(size=(len_ts, in_dim))
        fifo_test = amath.zeros((order, in_dim))
        for i in xrange(len_ts):
            self.assertTrue(amath.allclose(model.fifo.to_array(),
                                           fifo_test))
            model.learn_one_step(in_patterns[i])
            popped_in_pattern = model._update_state(in_patterns[i])
            if i < order:
                self.assertTrue(amath.allclose(popped_in_pattern,
                                               amath.zeros(in_dim)))
            else:
                self.assertTrue(amath.allclose(popped_in_pattern,
                                               in_patterns[i - order]))
            fifo_test[1:] = fifo_test[:-1]
            fifo_test[0] = in_patterns[i]


class VectorRegressionTestCaseNumpy(NumpyTestMixin,
                                    VectorRegressionTestCase,
                                    unittest.TestCase):
    pass


class VectorRegressionTestCaseCupy(CupyTestMixin,
                                   VectorRegressionTestCase,
                                   unittest.TestCase):
    pass


class VectorLogisticRegressionTestCase(object):
    """
    unit test for VectorLogisticRegression
    """

    def setUp(self):
        self.rate = 0.1

    def tearDown(self):
        pass

    def testGenerative(self):
        """
        testing minimal consistency in learning a sequence
        """
        print("VectorLogisticRegressionTestCase.testGenerative")
        in_dim = 3     # dimension of input sequence
        max_repeat = 10000
        for order in [1, 2]:
            for SGD in [AdaGrad]:
                model = VectorLogisticRegression(in_dim, in_dim, order, SGD())
                model.set_learning_rate(self.rate)
                i = tests.simple.test_binary_model(model, max_repeat)
                self.assertLess(i, max_repeat)

    def testDiscriminative(self):
        """
        testing minimal consistency in learning a sequence to an output
        """
        print("VectorLogisticRegressionTestCase.testDiscriminative")
        in_dim = 3
        out_dim = 2    # dimension of output sequence
        max_repeat = 10000
        for order in [1, 2]:
            for SGD in [AdaGrad]:
                model = VectorLogisticRegression(in_dim, out_dim, order, SGD())
                model.set_learning_rate(self.rate)
                i = tests.simple.test_binary_model(model, max_repeat)
                self.assertLess(i, max_repeat)


class VectorLogisticRegressionTestCaseNumpy(NumpyTestMixin,
                                            VectorLogisticRegressionTestCase,
                                            unittest.TestCase):
    pass


class VectorLogisticRegressionTestCaseCupy(CupyTestMixin,
                                           VectorLogisticRegressionTestCase,
                                           unittest.TestCase):
    pass


class MultiTargetVectorRegressionTestCase(object):
    """
    unit test for MultiTargetVectorRegression
    """

    def setUp(self):
        # learning rate
        self.rate = 0.01

    def testFifo(self):
        """
        testing fifo and _update_state method in MultiTargetVectorRegression
        """
        print("\n * testing fifo and update_state method "
              "in MultiTargetVectorRegression \n")
        in_dim = 3
        out_dims = [2, 4]
        SGDs = [AdaGrad(), AdaGrad()]
        order = 3

        len_ts = 10

        model = MultiTargetVectorRegression(in_dim, out_dims, SGDs, order)
        random = amath.random.RandomState(0)
        in_patterns = amath.random.uniform(size=(len_ts, in_dim))
        out_pattern_0 = amath.random.uniform(size=(len_ts, out_dims[0]))
        out_pattern_1 = amath.random.uniform(size=(len_ts, out_dims[1]))
        fifo_test = amath.zeros((order, in_dim))
        for i in xrange(len_ts):
            self.assertTrue(amath.allclose(model.layers[0].fifo.to_array(),
                                           fifo_test))
            model.learn_one_step([out_pattern_0[i], out_pattern_1[i]])
            popped_in_pattern = model._update_state(in_patterns[i])
            if i < order:
                self.assertTrue(amath.allclose(popped_in_pattern,
                                               amath.zeros(in_dim)))
            else:
                self.assertTrue(amath.allclose(popped_in_pattern,
                                               in_patterns[i - order]))
            fifo_test[1:] = fifo_test[:-1]
            fifo_test[0] = in_patterns[i]


class MultiTargetVectorRegressionTestCaseNumpy(
        NumpyTestMixin,
        MultiTargetVectorRegressionTestCase,
        unittest.TestCase):
    pass


class MultiTargetVectorRegressionTestCaseCupy(
        CupyTestMixin,
        MultiTargetVectorRegressionTestCase,
        unittest.TestCase):
    pass


if __name__ == "__main__":

    unittest.main()
