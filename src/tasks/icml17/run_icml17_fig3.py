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

"""
Script for running experiments for Figure 3 in our ICML 2017 paper.

python run_icml_fig3.py 6 0 false
python run_icml_fig3.py 6 1 false
python run_icml_fig3.py 6 1 true
python run_icml_fig3.py 8 0 false
python run_icml_fig3.py 8 1 false
python run_icml_fig3.py 8 1 true

After obtaining the results, run plot_icml17_fig3.py to make the figure.

.. seealso:: Takayuki Osogami, Hiroshi Kajino, and Taro Sekiyama, \
"Bidirectional learning for time-series models with hidden units", \
ICML 2017.
"""

__author__ = "Takayuki Osogami"


import pydybm.arraymath as amath
import pydybm.arraymath.dycupy as dycupy
# Uncomment the following line to run with cupy, which however
# slows down execution due to large overhead for instances
# of this size
# amath.setup(dycupy)
import argparse
import numpy as np
import os
import pickle
from copy import deepcopy
from pydybm.time_series.dybm import GaussianBernoulliDyBM
from pydybm.base.metrics import RMSE
from pydybm.base.generator import NoisySawtooth
from pydybm.base.sgd import AdaGrad


def experiment(period, std, delay, decay, Nh, repeat, bidirectional,
               sigma=0.01):
    """
    A run of experiment

    Parameters
    ----------
    period : int
        period of the wave
    std : float
        standard deviation of noise
    delay : int
        delay
    decay : list
        list of decay rates
    Nh : int
        number of hidden units
    repeat : int
        number of iterations of training
    bidirectional : boolean
        whether to train bidirectionally
    sigma : float
        std of random initialization
        0.01 is recommended in
        https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """

    """
    Prepare data generators
    """

    dim = 1                # dimension of the wave
    phase = amath.zeros(dim)  # initial phase of the wave

    # forward sequence
    wave = NoisySawtooth(0, period, std, dim, phase, False)
    wave.reset(seed=0)

    # backward sequence
    revwave = NoisySawtooth(0, period, std, dim, phase, True)
    revwave.reset(seed=1)

    """
    Prepare a Gaussian Bernoulli DyBM
    """

    Nv = dim      # number of visible units

    sgd = AdaGrad()
    dybm = GaussianBernoulliDyBM([delay, delay], [decay, decay], [Nv, Nh],
                                 [sgd, deepcopy(sgd)], sigma=sigma,
                                 insert_to_etrace="w_delay")
    dybm.layers[0].layers[1].SGD.set_learning_rate(0)
    dybm.layers[1].layers[1].SGD.set_learning_rate(0)

    """
    Learn
    """
    error = list()  # list of numpy array
    bi_end = 0.5
    bi_factor = 2
    for i in range(repeat):
        # update internal states by reading forward sequence
        wave.add_length(period)
        dybm.get_predictions(wave)

        if bidirectional and i % (bi_factor + 1) == 0 and bi_factor > 0 \
           and i < repeat * bi_end:
            # make a time-reversed DyBM
            dybm._time_reversal()

            # update internal states by reading backward sequence
            revwave.add_length(period)
            dybm.get_predictions(revwave)

            # learn backward sequence for one period
            revwave.add_length(period)
            dybm.learn(revwave, get_result=False)

            # make a non time-reversed DyBM
            dybm._time_reversal()
        else:
            # update internal states by reading forward sequence
            wave.add_length(period)
            dybm.get_predictions(wave)

            # learn forward sequence
            wave.add_length(period)
            result = dybm.learn(wave, get_result=True)

            if i % (bi_factor + 1) == bi_factor:
                rmse = RMSE(result["actual"], result["prediction"])
                rmse = amath.to_numpy(rmse)
                error.append(rmse)

    return error, dybm, wave


def get_filename(period, std, delay, decay, Nh, repeat,
                 bidirectional):
    filename = "saw_period" + str(period) + "_std" + str(std) \
               + "_delay" + str(delay) + "_decay" + str(decay) \
               + "_Nh" + str(Nh) + "_repeat" + str(repeat) \
               + "_bi" + str(bidirectional)
    return filename


def get_repeat(period):
    if period < 7:
        repeat = 1000
    else:
        repeat = 30000
    return repeat


if __name__ == "__main__":

    """
    Figure 3
    """

    parser = argparse.ArgumentParser(
        description='Evaluation of bidirectional training on sawtooth')
    parser.add_argument("period", help="Period: 6, 8, ...")
    parser.add_argument("Nh", help="Number of hidden units: 0, 1, ...")
    parser.add_argument("bidirectional", help="bidirectional: true, false")
    args = parser.parse_args()

    print(args)

    period = int(args.period)
    Nh = int(args.Nh)  # number of hidden units
    if args.bidirectional in ["true", "True"]:
        bidirectional = True
    else:
        bidirectional = False

    delay = 4
    decay = 0.0
    std = 0.01    # standard deviation of noise

    # run experiments

    directory = "sawtooth_results/"
    if not os.path.exists(directory):
        print("Creating directory " + directory)
        os.mkdir(directory)

    repeat = get_repeat(period)
    filename = get_filename(period, std, delay, decay, Nh, repeat,
                            bidirectional)
    print("Making " + filename)
    if not os.path.exists(directory + filename + ".npy"):
        error, dybm, wave = experiment(period, std, delay, [decay], Nh,
                                       repeat, bidirectional)
        # error is list()
        error = amath.array(error)
        error = amath.to_numpy(error)
        error.dump(directory + filename + ".npy")
        pickle.dump(dybm, open(directory + filename + ".pkl", "w"))
        pickle.dump(wave, open(directory + filename + "_wave.pkl", "w"))

