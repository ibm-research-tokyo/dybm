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

"""``numpy``-based implementation of multi-dimensional arrays.

"""

__author__ = "Taro Sekiyama"


import numpy
import scipy
import sklearn.metrics


def to_numpy(x):
    return x


array = numpy.array
empty = numpy.empty
zeros = numpy.zeros
ones = numpy.ones
arange = numpy.arange
eye = numpy.eye


# Attributes
ndim = numpy.ndim

# Mathematical operations on arrays or numbers
log = numpy.log
exp = numpy.exp
sqrt = numpy.sqrt
abs = numpy.abs
sign = numpy.sign
sin = numpy.sin
tanh = numpy.tanh
floor = numpy.floor
square = numpy.square


# Reduction operations on arrays
sum = numpy.sum
max = numpy.max
mean = numpy.mean
median = numpy.median
var = numpy.var
prod = numpy.prod
cond = numpy.linalg.cond
argmax = numpy.argmax
asarray = numpy.asarray


def root_mean_square_err(expect, correct):
    return numpy.sqrt(mean_squared_error(expect, correct))


# Matrix operations
dot = numpy.dot
transpose = numpy.transpose
tensordot = numpy.tensordot
multiply = numpy.multiply

maximum = numpy.maximum
minimum = numpy.minimum
concatenate = numpy.concatenate

diag = numpy.diag
allclose = numpy.allclose
outer = numpy.outer
inner = numpy.inner


def roll(a, shift):
    return numpy.roll(a, shift, axis=0)


# Constants
inf = numpy.inf
pi = numpy.pi
identity = numpy.identity
newaxis = numpy.newaxis


# Modules
random = numpy.random


# scipy functions
cho_factor = scipy.linalg.cho_factor
cho_solve = scipy.linalg.cho_solve
stats_multivariate_normal_logpdf = scipy.stats.multivariate_normal.logpdf


def linalg_solve(a, b):
    return scipy.linalg.solve(a, b, sym_pos=True)


# sklearn functions
mean_squared_error = sklearn.metrics.mean_squared_error
log_loss = sklearn.metrics.log_loss
kernel_metrics = sklearn.metrics.pairwise.kernel_metrics()
log_logistic = sklearn.utils.extmath.log_logistic


# advanced indexing
def assign_if_true(a, b, x):
    a[b] = x


# Matrix operations
from . import operations
op = operations

# FIFO data structures
from . import fifo

# DataQueue
from .data_queue import DataQueue
