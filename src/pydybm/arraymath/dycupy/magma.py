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

"""Fast implementation to solve :math:`Ax = B` using GPUs.
"""

__author__ = "Taro Sekiyama"


import ctypes
import functools


_lib = ctypes.CDLL('libmagma.so')
_lib.magma_init()


def _check_error(name, res, func, arguments):
    del func, arguments

    if res < 0:
        if res == -1:
            order = '1st'
        elif res == -2:
            order = '2nd'
        elif res == -3:
            order = '3rd'
        else:
            order = '{}th'.format(-res)
        raise RuntimeError('{} causes a run-time error at '
                           'the {} argument'.format(name, order))
    elif res > 0:
        raise RuntimeError('{} causes a run-time error '
                           'with info {}'.format(name, res))


def _magma_math_function(name, func, restype, argtypes,
                         argmap=lambda *args: args):
    func.restype = restype
    func.argtypes = argtypes
    func.errcheck = functools.partial(_check_error, name)

    def call(*args):
        args = argmap(*args)
        args += (ctypes.byref(ctypes.c_int()),)
        func(*args)

    return call


_lib.magma_uplo_const.restype = ctypes.c_int
_lib.magma_uplo_const.argtypes = (ctypes.c_char,)


dposv = _magma_math_function('magma_dposv_gpu', _lib.magma_dposv_gpu,
                             ctypes.c_int,
                             (ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p),
                             argmap=lambda uplo, *args:
                             (_lib.magma_uplo_const(uplo),) + args)


sposv = _magma_math_function('magma_sposv_gpu', _lib.magma_sposv_gpu,
                             ctypes.c_int,
                             (ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p),
                             argmap=lambda uplo, *args:
                             (_lib.magma_uplo_const(uplo),) + args)
