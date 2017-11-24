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

"""``cupy``-based implementation of multi-dimensional arrays.
"""

__author__ = "Taro Sekiyama"


import math
from chainer import cuda
import copy
import cupy
import numpy
import scipy
import sklearn.metrics
from . import random

try:
    from . import magma
    _magma_supported = True
except OSError:
    _magma_supported = False


memory_pool = cupy.cuda.MemoryPool()
cupy.cuda.set_allocator(memory_pool.malloc)


def to_numpy(x):
    return cuda.to_cpu(x) if isinstance(x, cupy.ndarray) \
        else numpy.array(x, copy=False)


array = cupy.array
empty = cupy.empty
zeros = cupy.zeros
ones = cupy.ones
arange = cupy.arange
eye = cupy.eye


# Attributes
def ndim(a):
    return a.ndim


# Mathematical operations on arrays or numbers
log = cupy.log
exp = cupy.exp
sqrt = cupy.sqrt
abs = cupy.abs
sign = cupy.sign
sin = cupy.sin
tanh = cupy.tanh
floor = cupy.floor
square = cupy.square


# Reduction operations on arrays
sum = cupy.sum
max = cupy.max
mean = cupy.mean
prod = cupy.prod
var = cupy.var
argmax = cupy.argmax
asarray = cupy.asarray


def median(a, **kwargs):
    return cuda.to_gpu(numpy.median(cuda.to_cpu(a), **kwargs))


def cond(a, **kwargs):
    return cuda.to_gpu(numpy.linalg.cond(cuda.to_cpu(a), **kwargs))


def root_mean_square_err(expect, correct):
    return cupy.sqrt(mean_squared_error(expect, correct))


# Matrix operations
dot = cupy.dot
transpose = cupy.transpose
tensordot = cupy.tensordot
multiply = cupy.multiply

maximum = cupy.maximum
minimum = cupy.minimum
concatenate = cupy.concatenate

diag = cupy.diag
outer = cupy.outer
inner = cupy.inner


def roll(a, shift):
    return cupy.roll(a, shift, axis=0)


def allclose(a, b, rtol=1e-05, atol=1e-08):
    return cupy.all(cupy.abs(a - b) < (atol + rtol * cupy.abs(b)))


# Constants
inf = numpy.inf
pi = numpy.pi
identity = cupy.identity
newaxis = cupy.newaxis


# # Modules
# random = cupy.random


# scipy functions
def cho_factor(a, **kwargs):
    c, lower = scipy.linalg.cho_factor(cuda.to_cpu(a), **kwargs)
    return (cuda.to_gpu(c), lower)


def cho_solve(a, b, **kwargs):
    c, lower = a
    return cuda.to_gpu(scipy.linalg.cho_solve((cuda.to_cpu(c), lower),
                                              cuda.to_cpu(b), **kwargs))


def linalg_solve_magma_(a, b, overwrite_b=True):
    assert a.flags.c_contiguous and b.flags.c_contiguous, \
        'input arrays should be c-contiguous'

    n, nrhs = b.shape if b.ndim == 2 else b.shape + (1,)
    x = b if overwrite_b else b.copy()
    if a.dtype == cupy.float64:
        posv = magma.dposv
    elif a.dtype == cupy.float32:
        posv = magma.sposv
    else:
        assert False, '{} is not supported'.format(a.dtype)

    posv("U", n, nrhs, a.data.ptr, n, x.data.ptr, n)
    return x


def linalg_solve_scipy_(a, b):
    return cuda.to_gpu(scipy.linalg.solve(cuda.to_cpu(a),
                                          cuda.to_cpu(b),
                                          sym_pos=True))


linalg_solve = linalg_solve_magma_ if _magma_supported else linalg_solve_scipy_


def stats_multivariate_normal_logpdf(x, **kwargs):
    for k in kwargs:
        kwargs[k] = to_numpy(kwargs[k])
    return scipy.stats.multivariate_normal.logpdf(cuda.to_cpu(x), **kwargs)


# sklearn functions
def mean_squared_error(y_true,
                       y_pred,
                       multioutput='uniform_average',
                       **kwargs):
    if multioutput == 'uniform_average':
        return cupy.mean((y_true - y_pred) ** 2.)
    else:
        y_true = to_numpy(y_true)
        y_pred = to_numpy(y_pred)
        return cuda.to_gpu(sklearn.metrics.mean_squared_error(
            y_true,
            y_pred,
            multioutput=multioutput,
            **kwargs
        ))


def log_loss(*args, **kwargs):
    raise


# kernels
def _rbf_kernel(x, y, gamma=None):
    xn, nx = x.shape
    _, ny = y.shape
    assert nx == ny, ('The number ({}) of columns of x must be the same as '
                      'the number ({}) of rows of y'.format(nx, ny))

    if gamma is None:
        gamma = 1.0 / xn

    xy = cupy.dot(x, y.transpose())
    x2 = (x * x).sum(axis=1)
    y2 = (y * y).sum(axis=1)

    return cupy.exp((x2[:, cupy.newaxis] - 2 * xy + y2) * -gamma)


kernel_metrics = copy.copy(sklearn.metrics.pairwise.kernel_metrics())


def wrap_kernel_metrics_(f):

    def wrapper(*args, **kwargs):
        args = map(to_numpy, args)
        for k in kwargs:
            kwargs[k] = to_numpy(kwargs[k])
        return cuda.to_gpu(f(*args, **kwargs))

    return wrapper


for k in kernel_metrics:
    kernel_metrics[k] = wrap_kernel_metrics_(kernel_metrics[k])

kernel_metrics['rbf'] = _rbf_kernel


_log_logistic_kernel = cupy.ElementwiseKernel(
    'T e, T x',
    'T o',
    'o = log(1.0 / (1.0 + pow(e, -x)))',
    'log_logistic_kernel'
)


def log_logistic(x):
    return _log_logistic_kernel(math.e, x)


# advanced indexing
_assign_if_true_kernel = cupy.ElementwiseKernel(
    'float64 a, bool b, float64 x',
    'float64 o',
    'o = b ? x : a',
    'assign_if_true_kernel'
)


def assign_if_true(a, b, x):
    _assign_if_true_kernel(a, b, x, a)


# Matrix operations
from . import operations
op = operations


# FIFO data structures
from . import fifo


# DataQueue
from .data_queue import DataQueue
