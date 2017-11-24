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
from tests.arraymath import NumpyTestMixin, CupyTestMixin
from pydybm.base.metrics import MSE, RMSE, baseline_RMSE


class metricsTestCase(object):
    """
    unit test for metrics
    """

    def setUp(self):
        self.L = 4
        self.N = 3

    def tearDown(self):
        pass

    def testMSE(self):
        np.random.seed(0)
        y = np.random.random((self.L, self.N))
        z = y + 1.0
        err = MSE(y, y)
        self.assertAlmostEqual(err, 0)
        err = MSE(y, z)
        self.assertAlmostEqual(err, self.N)

    def testRMSE(self):
        np.random.seed(0)
        y = np.random.random((self.L, self.N))
        z = y + 1.0
        err = RMSE(y, y)
        self.assertAlmostEqual(err, 0)
        err = RMSE(y, z)
        self.assertAlmostEqual(err, np.sqrt(self.N))

    def testBaseline_RMSE(self):
        np.random.seed(0)
        pattern = np.random.random(self.N)
        y = [pattern] * self.L
        err = baseline_RMSE(pattern, y)
        self.assertAlmostEqual(err,0)
        y = range(self.L)
        init_pred = -1
        err = baseline_RMSE(init_pred, y)
        self.assertAlmostEqual(err, 1)


class metricsTestCaseNumpy(NumpyTestMixin,
                           metricsTestCase,
                           unittest.TestCase):
    pass


class metricsTestCaseCupy(CupyTestMixin,
                          metricsTestCase,
                          unittest.TestCase):
    pass


if __name__ == "__main__":

    unittest.main()

