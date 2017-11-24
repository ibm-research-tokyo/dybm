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

"""``cupy``-based implementation of operations specific in DyBMs
"""

__author__ = "Taro Sekiyama"


import cupy

divide_by_pow_kernel_ = cupy.ElementwiseKernel(
    'float64 x, float64 y, float64 n',
    'float64 o',
    'o = x / pow(y, n)',
    'divide_by_pow'
)


def divide_by_pow(x, y, n):
    divide_by_pow_kernel_(x, y, n, x)
    return x


def divide_3d_by_1d_pow(x, y, n):
    return divide_by_pow(x, y, n)


vecreg_gradient_s_kernel_ = cupy.ElementwiseKernel(
    'float64 s, float64 x, float64 expected',
    'float64 o',
    'o = pow(x-expected, 2) / pow(s, 3) - 1.0 / s',
    'vecreg_gradient_s'
)


def vecreg_gradient_s(s, x, expected):
    return vecreg_gradient_s_kernel_(s, x, expected)


sgd_L1_regularization_kernel_ = cupy.ElementwiseKernel(
    'float64 variable, float64 delta, float64 strength, float64 th',
    'float64 o',
    'o = max(abs(variable + delta) - strength * th, 0.0) * '
    '    (variable + delta == 0 ? 0 : variable + delta > 0 ? 1 : -1)',
    'sgd_L1_regularization'
)


def sgd_L1_regularization(variable, delta, strength, th):
    if strength == 0.0:
        variable += delta
    else:
        sgd_L1_regularization_kernel_(variable, delta, strength, th, variable)
    return variable


rmsprop_get_delta_kernel_ = cupy.ElementwiseKernel(
    'float64 alpha, float64 first, float64 second, float64 delta',
    'float64 o',
    'o = alpha * first / (sqrt(second) + delta)',
    'rmsprop_get_delta'
)

rmsprop_get_delta = rmsprop_get_delta_kernel_
adagrad_get_delta = rmsprop_get_delta_kernel_


rmsprop_get_threshold_kernel_ = cupy.ElementwiseKernel(
    'float64 alpha, float64 second, float64 delta',
    'float64 o',
    'o = alpha / (sqrt(second) + delta)',
    'rmsprop_get_delta'
)

rmsprop_get_threshold = rmsprop_get_threshold_kernel_
adagrad_get_threshold = rmsprop_get_threshold_kernel_


rmsprop_update_state_kernel_ = cupy.ElementwiseKernel(
    'float64 gamma, float64 second, float64 grad',
    'float64 o',
    'o = gamma * second + (1-gamma) * pow(grad, 2)',
    'rmsprop_update_state'
)


def rmsprop_update_state(gamma, second, grad):
    rmsprop_update_state_kernel_(gamma, second, grad, second)
    return second


adagrad_update_state_kernel_ = cupy.ElementwiseKernel(
    'float64 second, float64 grad',
    'float64 o',
    'o = second + grad * grad',
    'adagrad_update_state'
)


def adagrad_update_state(second, grad):
    adagrad_update_state_kernel_(second, grad, second)
    return second


adagradplus_update_state_first_kernel_ = cupy.ElementwiseKernel(
    'float64 first, float64 beta, float64 grad',
    'float64 o',
    'o = beta * first + (1 - beta) * grad',
    'adagradplus_update_state_first'
)

adagradplus_update_state_second_kernel_ = cupy.ElementwiseKernel(
    'float64 second, float64 grad',
    'float64 o',
    'o = second + pow(grad, 2)',
    'adagradplus_update_state_second'
)


def adagradplus_update_state(first, second, beta, grad):
    adagradplus_update_state_first_kernel_(first, beta, grad, first)
    adagradplus_update_state_second_kernel_(second, grad, second)
    return first, second


sgd_L2_regularization_kernel_ = cupy.ElementwiseKernel(
    'float64 gradient, float64 strength, float64 variable',
    'float64 o',
    'o = gradient - strength * variable',
    'sgd_L2_regularization'
)


def sgd_L2_regularization(gradient, strength, variable):
    sgd_L2_regularization_kernel_(gradient, strength, variable, gradient)
    return gradient


adam_update_state_kernel_ = cupy.ElementwiseKernel(
    'float64 x, float64 a, float64 grad, float64 exponent',
    'float64 o',
    'o = a * x + (1 - a) * (pow(grad, exponent))',
    'adam_update_state'
)


def adam_update_state(first, second, beta, gamma, grad):
    adam_update_state_kernel_(first, beta, grad, 1, first)
    adam_update_state_kernel_(second, gamma, grad, 2, second)
    return first, second


adam_get_delta_kernel_ = cupy.ElementwiseKernel(
    'float64 first, float64 second,'
    'float64 alpha, float64 beta, float64 gamma,'
    'float64 epsilon, float64 step',
    'float64 o',
    'o = (first * alpha / (sqrt(second / (1 - pow(gamma, step))) + epsilon)) /'
    '    (1 - pow(beta, step))',
    'adam_get_delta'
)


def adam_get_delta(first, second, alpha, beta, gamma, epsilon, step):
    return adam_get_delta_kernel_(first, second, alpha,
                                  beta, gamma, epsilon, step)


adam_get_threshold_kernel_ = cupy.ElementwiseKernel(
    'float64 second,'
    'float64 alpha, float64 beta, float64 gamma,'
    'float64 epsilon, float64 step',
    'float64 o',
    'o = (alpha / (sqrt(second / (1 - pow(gamma, step))) + epsilon)) /'
    '    (1 - pow(beta, step))',
    'adam_get_threshold'
)


def adam_get_threshold(second, alpha, beta, gamma, epsilon, step):
    return adam_get_threshold_kernel_(second, alpha, beta,
                                      gamma, epsilon, step)


def update_e_trace(e_trace, decay_rates, in_pattern):
    return e_trace * decay_rates + in_pattern


def mult_2d_1d_to_3d(d2, d1):
    X, Y = d2.shape
    return d2.reshape(X, Y, 1) * d1


def mult_ijk_ij_to_ijk(d3, d2):
    X, Y = d2.shape
    return d3 * d2.reshape((X, Y, 1))


def mult_ijk_ij_to_jk(d3, d2):
    return mult_ijk_ij_to_ijk(d3, d2).sum(axis=0)
