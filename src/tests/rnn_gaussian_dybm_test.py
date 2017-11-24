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

""" Unit test for RNNGaussianDyBM """

__author__ = "Sakyasingha Dasgupta"


import unittest
import six

import tests.simple
from tests.arraymath import NumpyTestMixin, CupyTestMixin
import pydybm.arraymath as amath
from pydybm.time_series.dybm import LinearDyBM
from pydybm.time_series.rnn_gaussian_dybm import RNNGaussianDyBM, GaussianDyBM
from pydybm.base.sgd import RMSProp, AdaGrad


class RNNGaussianDyBMTestCase(object):
    """
    unit test for RNNGaussianDyBM
    """

    def setUp(self):
        self.max_repeat = 100000
        self.in_dim = 3     # dimension of input sequence
        self.out_dim = 2    # dimension of target sequence
        self.rnn_dim = 10
        self.sparsity = 0.1
        self.spectral_radius = 0.95
        self.leak = 1.0
        self.decay_rates = amath.array([0.2, 0.5, 0.8])

    def tearDown(self):
        pass

    def testGenerativeRNNGaussianDyBM(self):
        """
        testing minimal consistency in learning/generating a sequence
        """
        print("""
                ---------------------------------------------------------
                    test generative learning with RNNGaussianDyBM
                ---------------------------------------------------------
                """)
        print("\nDyBMTestCase.testGenerativeRNNGaussianDyBM")
        for delay in [0, 1, 3]:
            for SGD in [AdaGrad]:
                for GENERATOR in [True]:
                    print("\ndelay: %d" % delay)

                    model = RNNGaussianDyBM(self.in_dim, self.in_dim,
                                            self.rnn_dim, self.spectral_radius,
                                            self.sparsity, delay,
                                            self.decay_rates, self.leak)
                    model.set_learning_rate(0.01)
                    i = tests.simple.test_real_model(model, self.max_repeat,
                                                     GENERATOR)
                    self.assertLess(i, self.max_repeat)

    def testDiscriminativeRNNGaussianDyBM(self):
        """
        testing minimal consistency in learning a sequence to an output
        """
        print("""
                ---------------------------------------------------------
                    test discriminative learning with RNNGaussianDyBM
                ---------------------------------------------------------
                """)
        print("\nDyBMTestCase.testDiscriminativeRNNGaussianDyBM")

        for delay in [0, 1, 3]:
            for SGD in [AdaGrad]:
                for GENERATOR in [True]:
                    print("\ndelay: %d" % delay)
                    model = RNNGaussianDyBM(self.in_dim, self.out_dim,
                                            self.rnn_dim, self.spectral_radius,
                                            self.sparsity, delay,
                                            self.decay_rates, self.leak)
                    model.set_learning_rate(0.1)
                    i = tests.simple.test_real_model(model, self.max_repeat,
                                                     GENERATOR)
                    self.assertLess(i, self.max_repeat)


class RNNGaussianDyBMTestCaseNumpy(NumpyTestMixin,
                                   RNNGaussianDyBMTestCase,
                                   unittest.TestCase):
    pass


class RNNGaussianDyBMTestCaseCupy(CupyTestMixin,
                                  RNNGaussianDyBMTestCase,
                                  unittest.TestCase):
    pass


if __name__ == "__main__":

    unittest.main()
