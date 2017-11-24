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

"""``numpy``-based implementation of operations specific in DyBMs
"""

__author__ = "Taro Sekiyama"


import numpy


def divide_by_pow(x, y, n):
    return x / (y ** n)


def divide_3d_by_1d_pow(x, y, n):
    return divide_by_pow(x, y, n)


def vecreg_gradient_s(s, x, expected):
    return (x-expected)**2 / s**3 - 1.0 / s


def sgd_L1_regularization(variable, delta, strength, th):
    if strength == 0.0:
        variable += delta
    else:
        variable = sgd_L1_regularization_kernel(variable, delta, strength, th)
    return variable


def sgd_L1_regularization_kernel(variable, delta, strength, th):
    x = variable + delta
    y = numpy.abs(x) - strength * th
    z = numpy.maximum(y, numpy.zeros(y.shape))
    return numpy.sign(x) * z


def rmsprop_get_delta(alpha, first, second, delta):
    sqrt_second = numpy.sqrt(second) + delta
    return alpha * first / sqrt_second


def noisyrmsprop_get_delta(alpha, first, second, delta, noise):
    sqrt_second = numpy.sqrt(second) + delta
    return alpha * first / sqrt_second + noise


adagrad_get_delta = rmsprop_get_delta


def rmsprop_get_threshold(alpha, second, delta):
    sqrt_second = numpy.sqrt(second) + delta
    return alpha / sqrt_second


adagrad_get_threshold = rmsprop_get_threshold


noisyrmsprop_get_threshold = rmsprop_get_threshold


def rmsprop_update_state(gamma, second, grad):
    return gamma * second + (1-gamma) * grad**2


noisyrmsprop_update_state = rmsprop_update_state


def adagrad_update_state(second, g):
    return second + g**2


def adagradplus_update_state(first, second, beta, grad):
    return (beta * first + (1 - beta) * grad, second + grad**2)


def sgd_L2_regularization(gradient, strength, variable):
    return gradient - strength * variable


def adam_update_state(first, second, beta, gamma, grad):
    first = beta * first + (1 - beta) * grad
    second = gamma * second + (1 - gamma) * grad**2
    return first, second


def adam_get_delta(first, second, alpha, beta, gamma, epsilon, step):
    delta = alpha
    delta /= numpy.sqrt(second / (1 - gamma**step)) + epsilon
    delta *= first
    delta /= 1 - beta**step
    return delta


def adam_get_threshold(second, alpha, beta, gamma, epsilon, step):
    th = alpha
    th /= numpy.sqrt(second / (1 - gamma**step)) + epsilon
    th /= 1 - beta**step
    return th


def update_e_trace(e_trace, decay_rates, in_pattern):
    return e_trace * decay_rates + in_pattern


def mult_2d_1d_to_3d(d2, d1):
    X, Y = d2.shape
    return d2.reshape(X, Y, 1) * d1


def mult_ijk_ij_to_ijk(d3, d2):
    return numpy.einsum("ijk,ij->ijk", d3, d2)


def mult_ijk_ij_to_jk(d3, d2):
    return numpy.einsum("ijk,ij->jk", d3, d2)
