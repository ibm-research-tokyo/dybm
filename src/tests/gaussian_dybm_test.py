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
import numpy as np
import tests.simple
from tests.arraymath import NumpyTestMixin, CupyTestMixin
from pydybm.time_series.rnn_gaussian_dybm import GaussianDyBM
from pydybm.base.sgd import AdaGrad
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
        self.rate = 0.1    # learning rate

    def tearDown(self):
        pass

    def test_GenerativeGaussianDyBM(self):
        """ testing minimal consistency in learning a sequence
        """
        print("\nDyBMTestCase.testGenerativeGaussianDyBM")
        for delay in [1, 3]:
            for SGD in [AdaGrad]:
                model = GaussianDyBM(self.in_dim, self.in_dim, delay,
                                     SGD=SGD())
                model.set_learning_rate(self.rate)
                i = tests.simple.test_real_model(model, self.max_repeat,
                                                 True)
                self.assertLess(i, self.max_repeat)

    def test_DiscriminativeGaussianDyBM(self):
        """ testing minimal consistency in learning a sequence to an output
        """
        print("\nDyBMTestCase.testDiscriminativeGaussianDyBM")

        for delay in [1, 3]:
            for SGD in [AdaGrad]:
                model = GaussianDyBM(self.in_dim, self.out_dim, delay,
                                     SGD=SGD())
                model.set_learning_rate(self.rate)
                i = tests.simple.test_real_model(model, self.max_repeat,
                                                 True)
                self.assertLess(i, self.max_repeat)


class DyBMTestCaseNumpy(NumpyTestMixin, DyBMTestCase, unittest.TestCase):
    pass


class DyBMTestCaseCupy(CupyTestMixin, DyBMTestCase, unittest.TestCase):
    pass


if __name__ == "__main__":

    unittest.main()
