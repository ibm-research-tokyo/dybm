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
Script for making Figure 4 in our ICML 2017 paper.
Run run_icml17_fig4.py first.

.. seealso:: Takayuki Osogami, Hiroshi Kajino, and Taro Sekiyama, \
"Bidirectional learning for time-series models with hidden units", \
ICML 2017.
"""

__author__ = "Takayuki Osogami"


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot Figure 4")
    parser.add_argument("dataset", help="price or sunspot or climate")
    args = parser.parse_args()

    dataset = args.dataset

    baseline = {"climate": 2.51907116157,
                "price": 0.048838707735,
                "sunspot": 0.0770592078174}

    if dataset == "price":
        repeat = 10000
    else:
        repeat = 1000

    directory = dataset + "_test_results_largest" + str(repeat / 100) + "/"
    print("Reading", directory)
    ls = os.listdir(directory)

    delays = sorted(set([int(f.split("_")[3][5:])
                         for f in ls if f[-4:] == ".npy"]))
    Nhs = [4]

    best_err_nohidden = np.inf
    best_err_baseline = np.inf
    best_err_bi = np.inf
    for delay, Nh in product(delays, Nhs):
        print("\ndelay", delay, "Nh", Nh, "\t")
        plt.figure()

        # no hidden
        header = dataset + "_test_repeat" + str(repeat) + "_delay" \
                 + str(delay) + "_Nh0_bi0_end0.0"
        if dataset == "climate":
            header = header + "_std0.001"
        steps = np.load(directory + header + "_steps.npy")
        error = np.load(directory + header + "_test.npy")
        plt.plot(steps, error, label="No hidden", linestyle=":",
                 color="black")
        if min(error) < best_err_nohidden:
            print("*")
            best_err_nohidden = min(error)
        print(min(error))

        # baseline
        header = dataset + "_test_repeat" + str(repeat) + "_delay" \
                 + str(delay) + "_Nh" + str(Nh) + "_bi0_end0.0"
        if dataset == "climate":
            header = header + "_std0.001"
        steps = np.load(directory + header + "_steps.npy")
        error = np.load(directory + header + "_test.npy")
        plt.plot(steps, error, label="Baseline", linestyle="--",
                 color="black")
        if min(error) < best_err_baseline:
            print("*")
            best_err_baseline = min(error)
        print(min(error))

        # bidirectional
        ends = [0.25, 0.5, 1.0]
        for end in ends:
            header = dataset + "_test_repeat" + str(repeat) + "_delay" \
                     + str(delay) + "_Nh" + str(Nh) + "_bi2" + "_end" \
                     + str(end)
            if dataset == "climate":
                header = header + "_std0.001"
            steps = np.load(directory + header + "_steps.npy")
            error = np.load(directory + header + "_test.npy")
            if end == 1.0:
                plt.plot(steps, error, label="Bidirectional",
                         linestyle="-", color="black")
            else:
                plt.plot(steps, error, linestyle="-", color="black")
            if min(error) < best_err_bi:
                print("*")
                best_err_bi = min(error)
            print(min(error))

        plt.xlabel("Number of iterations", fontsize=18)
        plt.ylabel("Test RMSE", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18)
        if dataset == "sunspot":
            ylim = [0.069, baseline[dataset]]
        elif dataset == "price":
            ylim = [0.039, baseline[dataset]]
        elif dataset == "climate":
            ylim = [2.17, baseline[dataset]]
        plt.ylim(ylim)
        plt.xlim([0,max(steps)])
        figname = dataset + "_delay" + str(delay) + "_Nh" + str(Nh) + ".pdf"
        plt.savefig(figname, bbox_inches="tight", pad_inches=0.0)
