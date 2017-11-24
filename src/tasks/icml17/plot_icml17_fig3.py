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

"""
Script for making Figure 3 in our ICML 2017 paper.
Run run_icml17_fig3.py first.

.. seealso:: Takayuki Osogami, Hiroshi Kajino, and Taro Sekiyama, \
"Bidirectional learning for time-series models with hidden units", \
ICML 2017.
"""

__author__ = "Takayuki Osogami"


import pydybm.arraymath as amath
import pydybm.arraymath.dycupy as dycupy
# uncomment the following line if cupy is used in run_icml17_fig3.py
# amath.setup(dycupy)
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import pickle
from run_icml17_fig3 import get_repeat, get_filename


def get_style_label(bidirectional, Nh):
    if bidirectional:
        style = "-"
        label = u"Bidirectional"
    elif Nh > 0:
        style = "--"
        label = "Baseline"
    else:
        style = ":"
        label = "No hidden"
    return style, label


if __name__ == "__main__":

    """
    Figure 3
    """

    directory = "sawtooth_results/"
    delay = 4
    std = 0.01  # standard deviation of noise

    periods = [6, 8]
    decays = [0.0]
    directions = [True, False]
    Nhs = [0, 1]

    # Plot Figure 3 (a), (c)
    print("Making Figure 3(a), (c)")
    for (period, decay) in product(periods, decays):
        repeat = get_repeat(period)

        plt.figure(figsize=(6, 5))

        for bidirectional, Nh in product(directions, Nhs):
            if Nh == 0 and bidirectional:
                continue

            filename = get_filename(period, std, delay, decay, Nh,
                                    repeat, bidirectional)

            print("Loading " + directory + filename + ".*")
            error = np.load(directory + filename + ".npy")
            dybm = pickle.load(open(directory + filename + ".pkl", "r"))

            style, label = get_style_label(bidirectional, Nh)

            error = gaussian_filter1d(error, 50)
            steps = np.arange(len(error)) * 3
            plt.plot(steps, error, label=label, linestyle=style, color="black")
            print(delay, decay, Nh, bidirectional, "\t", error[-1])

        plt.legend(fontsize=17)
        plt.xlim([0, max(steps)])
        plt.ylim([0, 0.5])
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.xlabel("Number of iterations", fontsize=17)
        plt.ylabel("Training RMSE", fontsize=17)
        figname = "saw_period" + str(period) + "_delay" + str(delay) \
                  + "_decay" + str(decay)
        print("Writing " + figname)
        plt.savefig(figname + ".pdf", bbox_inches="tight", pad_inches=0.0)

    # Plot Figure 3 (b), (d)
    print("Making Figure 3(b), (d)")
    for (period, decay) in product(periods, decays):
        repeat = get_repeat(period)

        plt.figure(figsize=(6, 5))
        for bidirectional, Nh in product(directions, Nhs):
            if Nh == 0 and bidirectional:
                continue

            filename = get_filename(period, std, delay, decay, Nh, repeat,
                                    bidirectional)

            print("Loading " + directory + filename + ".pkl")
            dybm = pickle.load(open(directory + filename + ".pkl", "r"))
            print("Loading " + directory + filename + "_wave.pkl")
            wave = pickle.load(open(directory + filename + "_wave.pkl", "r"))
            wave.reset()
            wave.limit = period * 22
            wave.std = 0

            predictions = dybm.get_predictions(wave)
            target = wave.to_list()[-period * 2:]

            predictions = [amath.to_numpy(x)[0]
                           for x in predictions[-period * 2:]]
            print "predictions", predictions
            target = [amath.to_numpy(x)[0] for x in target]
            print "target", target

            style, label = get_style_label(bidirectional, Nh)

            plt.plot(predictions, label=label, color="black",
                     linestyle=style)
            if not bidirectional and Nh == 0:
                plt.plot(target, label="Target", color="red")

        plt.legend(loc="upper center", ncol=2, fontsize=17)
        plt.xlabel("Steps", fontsize=17)
        plt.ylabel("Target or predicted value", fontsize=17)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.xlim([0, period * 2 - 1])
        plt.ylim([0, 1.15])
        filename2 = "predict" + "_period" + str(period) + "_decay" + str(decay)
        plt.savefig(filename2 + ".pdf", bbox_inches="tight",
                    pad_inches=0.0)
