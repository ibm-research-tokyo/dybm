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
from itertools import product
import numpy as np

from tests.arraymath import NumpyTestMixin, CupyTestMixin
import pydybm.arraymath as amath
from pydybm.base.generator import *


class NoisyWaveTestCase(object):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testOneDimensional(self):
        eps = 1e-2
        period = 10
        for Wave in [NoisySin, NoisySawtooth]:
            for std in [0.0, 0.1, 1.0]:
                print("NoisyWaveTestCase.testOneDimensional " + str(Wave) +
                      " with std: " + str(std))
                length = period * 10
                dim = 1
                phase = None
                reverse = False

                if Wave == NoisySin:
                    angles = 2 * amath.pi * amath.arange(length) / period
                    target = amath.sin(angles)
                elif Wave == NoisySawtooth:
                    r = 1. * amath.arange(length) / period
                    target = r % 1
                else:
                    print(Wave)

                wave = Wave(length, period, std, dim, phase, reverse)

                sequence = list()
                for pattern in wave:
                    sequence.append(pattern[0])
                sequence = amath.array(sequence, dtype=float)

                diff = target - sequence

                rmse = amath.sqrt(amath.dot(diff, diff) / len(diff))

                print("RMSE: %f" % rmse)

                self.assertLessEqual(amath.abs(rmse - std), std * eps)


class NoisyWaveTestCaseNumpy(NumpyTestMixin,
                             NoisyWaveTestCase,
                             unittest.TestCase):
    pass


class NoisyWaveTestCaseCupy(CupyTestMixin,
                            NoisyWaveTestCase,
                            unittest.TestCase):
    pass


class UniformTestCase(object):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testOneDimensional(self):
        eps = 1e-2

        length = 10000
        dim = 1
        for low, high in product(range(3), range(3)):
            print("UniformTestCase.testOneDimensional Uniform[%d, %d]"
                  % (low, high))
            gen = Uniform(length, low, high, dim)

            sequence = list()
            for pattern in gen:
                sequence.append(pattern[0])
            sequence = amath.array(sequence, dtype=float)

            print("Mean: %f, Median: %f, Variance: %f"
                  % (amath.mean(sequence),
                     amath.median(sequence),
                     amath.var(sequence)))

            self.assertLessEqual(
                amath.abs(amath.mean(sequence) - 0.5*(low+high)),
                amath.abs(high-low)*eps)
            self.assertLessEqual(
                amath.abs(amath.median(sequence) - 0.5*(low+high)),
                amath.abs(high-low)*eps)
            self.assertLessEqual(
                amath.abs(amath.var(sequence) - (high-low)**2/12.),
                (high-low)**2*eps)

            sequence = amath.to_numpy(sequence)
            self.assertLessEqual(
                amath.abs(np.mean(sequence) - 0.5 * (low + high)),
                amath.abs(high - low) * eps)
            self.assertLessEqual(
                amath.abs(np.median(sequence) - 0.5 * (low + high)),
                amath.abs(high - low) * eps)
            self.assertLessEqual(
                amath.abs(np.var(sequence) - (high - low)**2 / 12.),
                (high - low)**2 * eps)


class UniformTestCaseNumpy(NumpyTestMixin, UniformTestCase, unittest.TestCase):
    pass


class UniformTestCaseCupy(CupyTestMixin, UniformTestCase, unittest.TestCase):
    pass


class SequenceTestCase(object):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testRandom(self):
        print("SequenceTestCase.testRandom")
        target = amath.random.random((10, 3))
        gen = SequenceGenerator(target.tolist())

        sequence = list()
        for pattern in gen:
            sequence.append(pattern)
        sequence = amath.array(sequence, dtype=float)

        diff = sequence - target

        self.assertEqual(amath.max(diff), 0)


class SequenceTestCaseNumpy(NumpyTestMixin,
                            SequenceTestCase,
                            unittest.TestCase):
    pass


class SequenceTestCaseCupy(CupyTestMixin,
                           SequenceTestCase,
                           unittest.TestCase):
    pass


if __name__ == "__main__":

    unittest.main()
