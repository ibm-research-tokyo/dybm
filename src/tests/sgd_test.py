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

import unittest


class TestCaseSGD(unittest.TestCase):

    def test_vSGD(self):
        from collections import defaultdict
        from pydybm.base.sgd import vSGD
        from pydybm.time_series.dybm import LinearDyBM
        from pydybm.base.generator import Uniform

        gen = Uniform(length=1000, low=0, high=1, dim=1)

        dybm = LinearDyBM(in_dim=1, delay=1, decay_rates=[], SGD=vSGD(hessian=defaultdict(lambda: 1)))
        dybm.learn(gen)

        self.assertEqual(float(dybm.variables['b']), 0.4932564368799297)


if __name__ == '__main__':
    unittest.main()
