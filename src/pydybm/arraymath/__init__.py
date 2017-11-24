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

"""Interface of array modules.

pydybm modules use multi-dimensional arrays via this interface.
Two implementations for this interface are given: numpy and cupy.
Users can enable DyBMs based on either by importing ``arraymath.dynumpy`` or
``arraymath.dycupy`` modules and giving it to ``setup`` function.
"""

__author__ = "Taro Sekiyama"


from . import dynumpy


def setup(lib):
    ''' Enable an array module globally.

    Parameters
    ----------
    lib: arraymath.dynumpy or arraymath.dycupy
    '''

    global to_numpy, array, empty, zeros, ones, arange, eye
    to_numpy = lib.to_numpy
    array = lib.array
    empty = lib.empty
    zeros = lib.zeros
    ones = lib.ones
    arange = lib.arange
    eye = lib.eye

    ''' Attributes '''
    global ndim
    ndim = lib.ndim

    ''' Mathematical operations on arrays or numbers '''
    global log, exp, sqrt, abs, sign, sin, tanh, floor
    log = lib.log
    exp = lib.exp
    sqrt = lib.sqrt
    abs = lib.abs
    sign = lib.sign
    sin = lib.sin
    tanh = lib.tanh
    floor = lib.floor

    ''' Reduction operations on arrays '''
    global sum, max, mean, median, var, prod, cond, argmax, asarray
    global root_mean_square_err
    sum = lib.sum
    max = lib.max
    mean = lib.mean
    median = lib.median
    var = lib.var
    prod = lib.prod
    cond = lib.cond
    argmax = lib.argmax
    asarray = lib.asarray
    root_mean_square_err = lib.root_mean_square_err

    ''' Matrix operations '''
    global dot, transpose, tensordot, multiply
    dot = lib.dot
    transpose = lib.transpose
    tensordot = lib.tensordot
    multiply = lib.multiply

    global maximum, minimum, concatenate
    maximum = lib.maximum
    minimum = lib.minimum
    concatenate = lib.concatenate

    global diag, roll, allclose, outer, inner
    diag = lib.diag
    roll = lib.roll
    allclose = lib.allclose
    outer = lib.outer
    inner = lib.inner

    ''' Constants '''
    global inf, pi, identity, newaxis
    inf = lib.inf
    pi = lib.pi
    identity = lib.identity
    newaxis = lib.newaxis

    ''' Modules '''
    global random
    random = lib.random

    ''' scipy functions '''
    global cho_factor, cho_solve, linalg_solve
    global stats_multivariate_normal_logpdf
    cho_factor = lib.cho_factor
    cho_solve = lib.cho_solve
    linalg_solve = lib.linalg_solve
    stats_multivariate_normal_logpdf = lib.stats_multivariate_normal_logpdf

    ''' sklearn functions '''
    global mean_squared_error, kernel_metrics, log_logistic
    mean_squared_error = lib.mean_squared_error
    kernel_metrics = lib.kernel_metrics
    log_logistic = lib.log_logistic

    ''' advanced indexing '''
    global assign_if_true
    assign_if_true = lib.assign_if_true

    ''' Matrix operations '''
    global op
    op = lib.op

    ''' FIFO data structures '''
    global FIFO
    FIFO = lib.fifo.FIFO

    ''' DataQueue '''
    global DataQueue
    DataQueue = lib.DataQueue


setup(dynumpy)
