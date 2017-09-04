# -*- coding: utf-8 -*-
""" DyBM_test """

__author__ = "Takayuki Osogami"
__copyright__ = "(C) Copyright IBM Corp. 2016"

import unittest
import numpy as np
import tests.simple
from tests.arraymath import NumpyTestMixin, CupyTestMixin
from pydybm.time_series.dybm import ComplexDyBM
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
        self.max_repeat = 10000
        self.in_dim = 3     # dimension of input sequence
        self.out_dim = 2    # dimension of target sequence
        self.rate = 0.01    # learning rate

    def tearDown(self):
        pass

    def test_ComplexDyBM(self):
        """ testing minimal consistency in learning a sequence to an output
        """
        print("\nDyBMTestCase.testComplexDyBM")
        dim = [2, 1, 2]
        delays = [2, 4, 3]
        rates = [[0.5], [0.3, 0.6], [0.5]]
        activations = ["linear", "sigmoid", "sigmoid"]

        for SGD in [AdaGrad]:
            for insert_to_etrace in ["w_delay", "wo_delay"]:
                print(SGD, insert_to_etrace)
                model = ComplexDyBM(delays, rates, activations, dim, SGD=SGD(),
                                    insert_to_etrace=insert_to_etrace)
                i = tests.simple.test_complex_model(model, self.max_repeat)
                self.assertLess(i, self.max_repeat)


class DyBMTestCaseNumpy(NumpyTestMixin, DyBMTestCase, unittest.TestCase):
    pass


class DyBMTestCaseCupy(CupyTestMixin, DyBMTestCase, unittest.TestCase):
    pass


if __name__ == "__main__":

    unittest.main()
