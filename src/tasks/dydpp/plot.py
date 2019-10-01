# (C) Copyright IBM Corp. 2019
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


"""Generate Figure 1 of Dynamic Determinantal Point Processes, AAAI-18.

"""

__author__ = "Takayuki Osogami"


import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate Figure 1 of Dynamic Determinantal Point Processes, AAAI-18"
    )
    parser.add_argument("data", help="in [JSB, Not, Pia, Mus]")
    args = parser.parse_args()

    logdir = "log/"
    data_name = args.data

    ls = os.listdir(logdir)
    files = [fname for fname in ls if fname.startswith(data_name)]

    allres = dict()
    for fname in files:
        with open(logdir+fname, "rb") as f:
            result = pickle.load(f)
        for key in result:
            if key in allres:
                print("exists", key)
            else:
                allres[key] = deepcopy(result[key])

    rangeD = sorted(set([key[0] for key in allres.keys()]))
    rangeK = sorted(set([key[1] for key in allres.keys()]))
    print(rangeD)

    linestyles = ["-", "--", ":", "-."]
    K0 = {"JSB": 4,
          "Not": 9,
          "Pia": 12,
          "Mus": 15}
    fig = plt.figure(figsize=(3, 3))
    x = [0] + rangeD
    for K in rangeK:
        if K % K0[data_name] != 0:
            continue
        mul = K // K0[data_name]
        if data_name == "JSB":
            if mul < 2 or mul > 4:
                continue
        else:
            if mul < 1 or mul > 3:
                continue
        y = [allres[(rangeD[0], K, 0, 0.0)]["DPP"]["testLL"]]
        y += [allres[(D, K, 0, 0.0)]["DyDPP"]["testLL"] for D in rangeD]
        plt.plot(x, y,
                 color="k",
                 linestyle=linestyles[mul-1],
                 label=u"$K=$"+str(mul)+u"$K_0$")
    plt.xlabel(u"$D$", fontsize=12)
    plt.ylabel("Test log likelihood", fontsize=12)
    plt.xticks(x)
    plt.xlim([0, 3])
    plt.legend()
    filename = data_name + ".pdf"
    fig.savefig(filename, bbox_inches="tight")
    print("Created", filename)
