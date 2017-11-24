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

"""``cupy``-based implementation of the random module
"""

__author__ = "Taro Sekiyama"


import numpy.random as r
import cupy as cp


def _to_gpu(a):
    arr = cp.empty_like(a)
    arr.set(a)
    return arr


class RandomState:
    def __init__(self, seed):
        self._random = r.RandomState(seed)

    def uniform(self, low=0.0, high=1.0, size=None):
        return _to_gpu(self._random.uniform(low=low, high=high, size=size))

    def normal(self, loc=0.0, scale=1.0, size=None):
        return _to_gpu(self._random.normal(loc=loc, scale=scale, size=size))

    def get_state(self):
        return self._random.get_state()

    def set_state(self, *args):
        return self._random.set_state(*args)

    def rand(self, *args):
        return _to_gpu(self._random.rand(*args))


seed = r.seed


def normal(loc=0.0, scale=1.0, size=None):
    return _to_gpu(r.normal(loc=loc, scale=scale, size=size))


def uniform(low=0.0, high=1.0, size=None):
    return _to_gpu(r.uniform(low=low, high=high, size=size))


def rand(*args):
    return _to_gpu(r.rand(*args))


def randn(*args):
    return _to_gpu(r.randn(*args))


def random(size=None):
    return _to_gpu(r.random(size=size))
