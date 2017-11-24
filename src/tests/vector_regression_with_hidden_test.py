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

""" Unittest for ``VectorRegressionWithHidden`` """

__author__ = "Hiroshi Kajino"
__version__ = "0.1"
__date__ = "Nov 11, 2016"

import scipy as sp
import unittest
from copy import deepcopy
from six.moves import xrange
import tests.simple
from tests.arraymath import NumpyTestMixin, CupyTestMixin
from pydybm.time_series.vector_regression import VectorRegressionWithHidden, sigmoid
from pydybm.base.sgd import AdaGrad
import pydybm.arraymath as amath


class VectorRegressionWithHiddenTest(object):
    """ unit test for VectorRegressionWithHidden
    """

    def setUp(self):
        self.in_dim = 4
        self.out_dim = 2
        self.dim_hidden = 1
        self.order = 3
        self.model = VectorRegressionWithHidden(in_dim=self.in_dim,
                                                out_dim=self.out_dim,
                                                dim_hidden=self.dim_hidden,
                                                order=self.order, SGD=AdaGrad(),
                                                L1=0., L2=0., use_bias=True,
                                                sigma=0.1)
        pass

    def tearDown(self):
        pass

    def test_init(self):
        """ check initialization
        """
        self.assertTrue(self.model.variables["U"].shape ==
                        (self.order, self.in_dim, self.dim_hidden))
        self.assertTrue(self.model.variables["V"].shape ==
                        (self.dim_hidden, self.out_dim))

    def test_u_tilde(self):
        """ check _get_u_tilde
        """
        in_patterns = [amath.array([1, 0, 0, 0]),
                       amath.array([0, 1, 0, 0]),
                       amath.array([0, 0, 1, 0])]
        self.model._update_state(in_patterns[2])
        self.model._update_state(in_patterns[1])
        self.model._update_state(in_patterns[0])
        u_tilde_test = amath.zeros(self.dim_hidden)
        for d in xrange(self.order):
            u_tilde_test += self.model.variables["U"][d, d, :]
        u_tilde_test += self.model.variables["b_h"]
        self.assertTrue(amath.allclose(u_tilde_test, self.model._get_u_tilde()))
        pass

    def test_mu(self):
        """ check mu given by get_conditional_negative_energy
        """
        in_patterns = [amath.array([1, 0, 0, 0]),
                       amath.array([0, 1, 0, 0]),
                       amath.array([0, 0, 1, 0])]
        self.model._update_state(in_patterns[2])
        self.model._update_state(in_patterns[1])
        self.model._update_state(in_patterns[0])
        fifo_array = self.model.fifo.to_array()
        self.assertTrue(amath.allclose(fifo_array[0], in_patterns[0]))
        self.assertTrue(amath.allclose(fifo_array[1], in_patterns[1]))
        self.assertTrue(amath.allclose(fifo_array[2], in_patterns[2]))
        mu_test = deepcopy(self.model.variables["b"])
        u_tilde = self.model._get_u_tilde()
        # naive implementation of sigmoid
        sig_u_tilde = sigmoid(u_tilde)
        for d in xrange(self.order):
            mu_test += self.model.variables["W"][d, d, :]
        mu_test += sig_u_tilde.dot(self.model.variables["V"])
        self.assertTrue(amath.allclose(
            self.model._get_conditional_negative_energy(), mu_test))
        pass

    def test_obj(self):
        """ test run _get_obj
        """
        in_patterns = [amath.array([1, 0, 0, 0]),
                       amath.array([0, 1, 0, 0]),
                       amath.array([0, 0, 1, 0])]
        self.model._update_state(in_patterns[2])
        self.model._update_state(in_patterns[1])
        self.model._update_state(in_patterns[0])
        out_pattern = amath.array([2.0, 1.0])
        print("\n * obj = {}".format(self.model._get_obj(out_pattern)))

    def test_grad(self):
        """ test run _get_gradient
        """
        in_patterns = [amath.array([1, 0, 0, 0]),
                       amath.array([0, 1, 0, 0]),
                       amath.array([0, 0, 1, 0])]
        self.model._update_state(in_patterns[2])
        self.model._update_state(in_patterns[1])
        self.model._update_state(in_patterns[0])
        out_pattern = amath.array([2.0, 1.0])
        grad = self.model._get_gradient(out_pattern)
        self.assertTrue(grad["b"].shape == (self.out_dim,))
        self.assertTrue(grad["W"].shape ==
                        (self.order, self.in_dim, self.out_dim))
        self.assertTrue(grad["U"].shape ==
                        (self.order, self.in_dim, self.dim_hidden))
        self.assertTrue(grad["V"].shape == (self.dim_hidden, self.out_dim))

    def testGenerative(self):
        """
        testing minimal consistency in learning a sequence
        """
        max_repeat = 100000
        self.model.set_learning_rate(0.01)
        i = tests.simple.test_real_model(self.model, max_repeat)
        self.assertLess(i, max_repeat)


class VectorRegressionWithHiddenTestNumpy(NumpyTestMixin,
                                          VectorRegressionWithHiddenTest,
                                          unittest.TestCase):
    pass


class VectorRegressionWithHiddenTestCupy(CupyTestMixin,
                                         VectorRegressionWithHiddenTest,
                                         unittest.TestCase):
    pass


if __name__ == "__main__":

    unittest.main()
