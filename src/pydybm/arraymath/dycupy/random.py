"""``cupy``-based implementation of the random module
"""

__author__ = "Taro Sekiyama"
__copyright__ = "(C) Copyright IBM Corp. 2016"


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
