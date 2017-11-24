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

""" Unit test for FunctionalDyBM """

__author__ = "Hiroshi Kajino"
__version__ = "1.0"
__date__ = "2nd Aug 2016"

import argparse
import unittest
import sys
import time
import os

from tests.arraymath import NumpyTestMixin, CupyTestMixin
import pydybm.arraymath as amath
from pydybm.time_series.functional_dybm import FunctionalDyBM, FuncLinearDyBM
from six.moves import xrange


def _pattern_func(loc, i):
    return (amath.sin(loc.sum(axis=1) + i / 10.0)
            + 0.00001 * amath.random.randn(loc.shape[0]))


class FunctionalDyBMTest(object):
    """ unit test for FunctionalDyBM
    """

    def setUp(self):
        self.dim = 2
        self.n_obs = 20
        self.n_anc = 5
        self.low = 0.0
        self.high = 1.0
        self.delay = 3
        self.decay_rates = [0.9, 0.2]
        self.n_elg = len(self.decay_rates)
        self.rand_seed = 0
        pass

    def tearDown(self):
        pass

    def test_init(self):
        print("""
        ---------------------------------------------
            test initialization of NeuralFieldDyBM
        ---------------------------------------------
        """)
        anc_test = amath.random.uniform(self.low, self.high,
                                        (self.n_anc, self.dim))

        model = FunctionalDyBM(
            self.dim, anc_test, self.delay, self.decay_rates)

        # check anchors
        amath.random.seed(self.rand_seed)
        self.assertTrue(amath.allclose(model.anc_basis, anc_test))

        # check the shapes
        self.assertTrue(model.variables["W"].shape ==
                        (self.delay - 1, self.n_anc, self.n_anc))
        self.assertTrue(model.variables["V"].shape ==
                        (self.n_elg, self.n_anc, self.n_anc))
        self.assertTrue(model.variables["b"].shape == (self.n_anc,))

        # check kernel
        X1 = amath.random.randn(10, self.dim)
        X2 = amath.random.randn(10, self.dim)
        rbf = amath.kernel_metrics['rbf'](X1, X2, gamma=1.0)
        self.assertTrue(amath.allclose(model.ker(X1, X2), rbf))

        # check anchor points
        self.assertTrue(model.anc_basis.shape == (self.n_anc, self.dim))

        # check fifo and e_trace
        self.assertTrue(model.fifo.shape == (self.delay - 1, self.n_anc))
        self.assertTrue(model.e_trace.shape == (self.n_elg, self.n_anc))
        return 0

    def test_update_state(self):
        print("""
        ---------------------------------
            test updating internal states
        ---------------------------------
        """)
        anc_test = amath.random.uniform(self.low, self.high,
                                        (self.n_anc, self.dim))
        model = FunctionalDyBM(
            self.dim, anc_test, self.delay, self.decay_rates)
        loc = amath.random.uniform(self.low, self.high, (self.n_obs, self.dim))

        for i in range(10):
            pattern = amath.zeros(self.n_obs)
            pattern[i % self.n_obs] = 1
            fifo_old = model.fifo[:-1]
            fifo_last = model.fifo[-1]
            e_trace_old = model.e_trace

            model._update_state(pattern, loc)

            # check fifo update
            self.assertTrue(amath.allclose(model.fifo[1:], fifo_old))

            # check e_trace update
            e_trace_new = amath.zeros((self.n_elg, self.n_anc)) + fifo_last
            for k in xrange(e_trace_new.shape[0]):
                e_trace_new[k, :] = e_trace_new[k, :] \
                    + self.decay_rates[k] * e_trace_old[k, :]
            self.assertTrue(amath.allclose(model.e_trace, e_trace_new))

            # check mu_anc update
            self.assertTrue(amath.allclose(model.mu, amath.zeros(self.n_anc)))

        print("mu(P) = {}".format(model.mu))
        return 0

    def test_learn_static_signal_static_observation(self):
        print("""
        ---------------------------------------------------------
            test learning
            using static signal and static observation points
        ---------------------------------------------------------
        """)
        MAX_ITER = 1000
        loc = amath.random.uniform(self.low, self.high, (self.n_obs, self.dim))
        model = FunctionalDyBM(self.dim, loc, self.delay, self.decay_rates,
                               noise_var=0.01,
                               ker_paras={"ker_type": "rbf", "gamma": 1.0},
                               learning_rate=0.00001)
        pattern = amath.ones(self.n_obs)

        for i in xrange(MAX_ITER):
            if i % 100 == 0:
                print("step {}:\t log-pdf = {} \t RSME = {}".format(
                    i, model.get_LL(pattern, loc),
                    model.compute_RMSE(pattern, loc)))
            model.learn_one_step(pattern, loc)

        print("\npattern = {}".format(pattern))
        print("pred = {}".format(model.predict_next(loc)))
        return 0

    def test_learning_static_signal_dynamic_observation(self):
        print("""
        ---------------------------------------------------------
            test learning
            using static signal and dynamic observation points
        ---------------------------------------------------------
        """)
        MAX_ITER = 1000
        anc_test = amath.random.uniform(self.low, self.high,
                                        (self.n_anc, self.dim))
        model = FunctionalDyBM(self.dim, anc_test, self.delay, self.decay_rates,
                               noise_var=0.1,
                               ker_paras={"ker_type": "rbf", "gamma": 1.0},
                               learning_rate=0.0001)
        for i in xrange(MAX_ITER):
            loc = amath.random.uniform(
                self.low, self.high, (self.n_obs, self.dim))
            pattern = _pattern_func(loc, 0)
            if i % 100 == 0:
                print("step {}:\t log-pdf = {} \t RSME = {}".format(
                    i, model.get_LL(pattern, loc),
                    model.compute_RMSE(pattern, loc)))
            model.learn_one_step(pattern, loc)

        loc = amath.random.uniform(self.low, self.high, (self.n_obs, self.dim))
        pattern = _pattern_func(loc, 0)
        print("\npattern = {}".format(pattern))
        print("pred = {}".format(model.predict_next(loc)))
        return 0

    def test_learning_dynamic_signal_static_observation(self):
        print("""
        ---------------------------------------------------------
            test learning
            using dynamic signal and static observation points
        ---------------------------------------------------------
        """)
        MAX_ITER = 1000
        loc = amath.random.uniform(self.low, self.high, (self.n_obs, self.dim))
        model = FunctionalDyBM(self.dim, loc, self.delay, self.decay_rates,
                               noise_var=0.01,
                               ker_paras={"ker_type": "rbf", "gamma": 1.0},
                               learning_rate=0.0001)

        for i in xrange(MAX_ITER):
            pattern = _pattern_func(loc, i)
            if i % 100 == 0:
                print("step {}:\t log-pdf = {} \t RSME = {}".format(
                    i, model.get_LL(pattern, loc),
                    model.compute_RMSE(pattern, loc)))
            model.learn_one_step(pattern, loc)

        pattern = _pattern_func(loc, MAX_ITER)
        print("\npattern = {}".format(pattern))
        print("pred = {}".format(model.predict_next(loc)))
        return 0

    def test_learning_dynamic_signal_dynamic_observation(self):
        print("""
        ---------------------------------------------------------
            test learning
            using dynamic signal and dynamic observation points
        ---------------------------------------------------------
        """)
        MAX_ITER = 1000
        anc_test = amath.random.uniform(self.low, self.high,
                                        (self.n_anc, self.dim))
        model = FunctionalDyBM(self.dim, anc_test,
                               self.delay, self.decay_rates,
                               noise_var=0.01,
                               ker_paras={"ker_type": "rbf", "gamma": 0.5},
                               learning_rate=0.001)

        for i in xrange(MAX_ITER):
            loc = amath.random.uniform(
                self.low, self.high, (self.n_obs, self.dim))
            pattern = _pattern_func(loc, i)
            if i % 100 == 0:
                print("step {}:\t log-pdf = {} \t RSME = {}".format(
                    i, model.get_LL(pattern, loc),
                    model.compute_RMSE(pattern, loc)))
            model.learn_one_step(pattern, loc)

        loc = amath.random.uniform(self.low, self.high, (self.n_obs, self.dim))
        pattern = _pattern_func(loc, MAX_ITER)
        print("\npattern = {}".format(pattern))
        print("pred = {}".format(model.predict_next(loc)))
        return 0

    def testFuncLinearDyBM(self):
        print("""
        -------------------------------------
            test FuncLinearDyBM
        -------------------------------------
        """)
        MAX_ITER = 1000
        anc_test = amath.random.uniform(self.low, self.high,
                                        (self.n_anc, self.dim))
        model = FuncLinearDyBM(self.dim, anc_test,
                               self.delay, self.decay_rates, noise_var=0.01,
                               ker_paras={"ker_type": "rbf", "gamma": 0.1},
                               insert_to_etrace="wo_delay",
                               learning_rate=0.001)
        for i in xrange(MAX_ITER):
            loc = amath.random.uniform(
                self.low, self.high, (self.n_obs, self.dim))
            pattern = _pattern_func(loc, i)
            if i % 100 == 0:
                print("step {}:\t RSME = {}".format(
                    i, model.compute_RMSE(pattern, loc)))
            model.learn_one_step(pattern, loc)

        loc = amath.random.uniform(self.low, self.high, (self.n_obs, self.dim))
        pattern = _pattern_func(loc, MAX_ITER)
        print("\npattern = {}".format(pattern))
        print("pred = {}".format(model.predict_next(loc)))
        return 0

    def testFuncLinearDyBMWithDelay(self):
        print("""
        -------------------------------------
            test FuncLinearDyBM with delay
        -------------------------------------
        """)
        MAX_ITER = 1000
        anc_test = amath.random.uniform(self.low, self.high,
                                        (self.n_anc, self.dim))
        model = FuncLinearDyBM(self.dim, anc_test,
                               self.delay, self.decay_rates, noise_var=0.01,
                               ker_paras={"ker_type": "rbf", "gamma": 0.1},
                               insert_to_etrace="w_delay", learning_rate=0.001)
        for i in xrange(MAX_ITER):
            loc = amath.random.uniform(
                self.low, self.high, (self.n_obs, self.dim))
            pattern = _pattern_func(loc, i)
            if i % 100 == 0:
                print("step {}:\t RSME = {}".format(
                    i, model.compute_RMSE(pattern, loc)))
            model.learn_one_step(pattern, loc)

        loc = amath.random.uniform(self.low, self.high, (self.n_obs, self.dim))
        pattern = _pattern_func(loc, MAX_ITER)
        print("\npattern = {}".format(pattern))
        print("pred = {}".format(model.predict_next(loc)))
        return 0

    def test_learning_FAR(self):
        print("""
        ---------------------------------------------------------
            test learning: Functional Autoregression
            using static signal and static observation points
        ---------------------------------------------------------
        """)
        MAX_ITER = 1000
        loc = amath.random.uniform(self.low, self.high, (self.n_obs, self.dim))
        model = FunctionalDyBM(self.dim, loc, self.delay, [], noise_var=0.01,
                               ker_paras={"ker_type": "rbf", "gamma": 10.0},
                               learning_rate=0.0001)
        pattern = amath.ones(self.n_obs)

        for i in xrange(MAX_ITER):
            if i % 100 == 0:
                print("step {}:\t log-pdf = {} \t RSME = {}".format(
                    i, model.get_LL(pattern, loc),
                    model.compute_RMSE(pattern, loc)))
            model.learn_one_step(pattern, loc)

        print("\npattern = {}".format(pattern))
        print("pred = {}".format(model.predict_next(loc)))
        return 0

    def test_different_anc_points(self):
        print("""
        ---------------------------------------------------------
            test different anchor points
        ---------------------------------------------------------
        """)
        MAX_ITER = 1000
        anc_basis = amath.random.uniform(self.low, self.high,
                                         (self.n_anc, self.dim))
        anc_data = amath.random.uniform(self.low, self.high,
                                        (self.n_anc, self.dim))
        model = FunctionalDyBM(self.dim, (anc_basis, anc_data), self.delay,
                               self.decay_rates, noise_var=0.01,
                               ker_paras={"ker_type": "rbf", "gamma": 0.5},
                               learning_rate=0.001)

        for i in xrange(MAX_ITER):
            loc = amath.random.uniform(
                self.low, self.high, (self.n_obs, self.dim))
            pattern = _pattern_func(loc, i)
            if i % 100 == 0:
                print("step {}:\t log-pdf = {} \t RSME = {}".format(
                    i, model.get_LL(pattern, loc),
                    model.compute_RMSE(pattern, loc)))
            model.learn_one_step(pattern, loc)

        loc = amath.random.uniform(self.low, self.high, (self.n_obs, self.dim))
        pattern = _pattern_func(loc, MAX_ITER)
        print("\npattern = {}".format(pattern))
        print("pred = {}".format(model.predict_next(loc)))
        return 0

    def test_learn_anc_points(self):
        print("""
        ---------------------------------------------------------
            test learn anc points
        ---------------------------------------------------------
        """)
        MAX_ITER = 1000
        anc_basis = amath.random.uniform(self.low, self.high,
                                         (self.n_anc, self.dim))
        anc_data = amath.random.uniform(self.low, self.high,
                                        (self.n_anc, self.dim))
        model = FunctionalDyBM(self.dim, (anc_basis, anc_data), self.delay,
                               self.decay_rates, noise_var=0.01,
                               ker_paras={"ker_type": "rbf", "gamma": 0.5},
                               learning_rate=0.001, learn_anc_basis=True)

        for i in xrange(MAX_ITER):
            loc = amath.random.uniform(
                self.low, self.high, (self.n_obs, self.dim))
            pattern = _pattern_func(loc, i)
            if i % 100 == 0:
                print("step {}:\t log-pdf = {} \t RSME = {}".format(
                    i, model.get_LL(pattern, loc),
                    model.compute_RMSE(pattern, loc)))
            model.learn_one_step(pattern, loc)

        loc = amath.random.uniform(self.low, self.high, (self.n_obs, self.dim))
        pattern = _pattern_func(loc, MAX_ITER)
        print("\npattern = {}".format(pattern))
        print("pred = {}".format(model.predict_next(loc)))
        return 0

    def test_learn_sequence(self):
        print("""
        ---------------------------------------------
            test learn in FunctionalDyBM
        ---------------------------------------------
        """)
        MAX_ITER = 1000
        anc_test = amath.random.uniform(self.low, self.high,
                                        (self.n_anc, self.dim))
        model = FunctionalDyBM(self.dim, anc_test, self.delay, self.decay_rates,
                               noise_var=0.01,
                               ker_paras={"ker_type": "rbf", "gamma": 0.5},
                               learning_rate=0.001)
        pattern_seq = []
        loc_seq = []
        for i in xrange(MAX_ITER):
            loc = amath.random.uniform(
                self.low, self.high, (self.n_obs, self.dim))
            loc_seq.append(loc)
            pattern = _pattern_func(loc, i)
            pattern_seq.append(pattern)
        model.learn(pattern_seq, loc_seq)
        loc = amath.random.uniform(self.low, self.high, (self.n_obs, self.dim))
        pattern = _pattern_func(loc, MAX_ITER)
        print("\npattern = {}".format(pattern))
        print("pred = {}".format(model.predict_next(loc)))
        return 0

    def test_learn_sequence_FuncLinearDyBM(self):
        print("""
        ---------------------------------------------
            test learn in FuncLinearDyBM
        ---------------------------------------------
        """)
        MAX_ITER = 1000
        anc_test = amath.random.uniform(self.low, self.high,
                                        (self.n_anc, self.dim))
        model = FuncLinearDyBM(self.dim, anc_test,
                               self.delay, self.decay_rates, noise_var=0.01,
                               ker_paras={"ker_type": "rbf", "gamma": 0.1},
                               insert_to_etrace="wo_delay", learning_rate=0.001)
        pattern_seq = []
        loc_seq = []
        for i in xrange(MAX_ITER):
            loc = amath.random.uniform(
                self.low, self.high, (self.n_obs, self.dim))
            loc_seq.append(loc)
            pattern = _pattern_func(loc, i)
            pattern_seq.append(pattern)
        model.learn(pattern_seq, loc_seq)
        loc = amath.random.uniform(self.low, self.high, (self.n_obs, self.dim))
        pattern = _pattern_func(loc, MAX_ITER)
        print("\npattern = {}".format(pattern))
        print("pred = {}".format(model.predict_next(loc)))
        return 0


class FunctionalDyBMTestNumpy(NumpyTestMixin,
                              FunctionalDyBMTest,
                              unittest.TestCase):
    pass


class FunctionalDyBMTestCupy(CupyTestMixin,
                             FunctionalDyBMTest,
                             unittest.TestCase):
    pass


if __name__ == "__main__":

    unittest.main()
