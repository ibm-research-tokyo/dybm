# (C) Copyright IBM Corp. 2015
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

__author__ = "Takayuki Osogami"


import networkx as nx
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import matplotlib.cm as cmx
import matplotlib.colors as colors


def read(filename):
    bias = dict()
    delay = dict()
    U = dict()
    V = dict()
    print "reading", filename
    f = open(filename)
    for line in f:
        w = line.strip().strip(",").split(",")
        if w[0] == "bias":
            node = int(w[1])
            b = float(w[2])
            bias[node] = b
        if w[0] == "delay":
            pre = int(w[1])
            post = int(w[2])
            d = int(w[3])
            delay[(pre, post)] = d
        if w[0] == "U":
            pre = int(w[1])
            post = int(w[2])
            U[(pre, post)] = [float(x) for x in w[3:]]
        if w[0] == "V":
            pre = int(w[1])
            post = int(w[2])
            V[(pre, post)] = [float(x) for x in w[3:]]
    f.close()
    return bias, delay, U, V


def drawDyBM(G, ax):
    W = 4
    H = 2

    # prepare color map
    cNorm = colors.Normalize(vmin=-1, vmax=1)
    sMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('coolwarm'))
    # drawing nodes
    for n in G:
        color = sMap.to_rgba(G.node[n]["bias"])
        pos = [G.node[n]["pos"][0] * W, G.node[n]["pos"][1] * H]
        c = Circle(pos, radius=0.3, alpha=1.0, color=color)
        ax.add_patch(c)
        G.node[n]['patch'] = c

    # drawing edges
    seen = dict()
    for (u, v, d) in G.edges(data=True):
        n1 = G.node[u]['patch']
        n2 = G.node[v]['patch']
        rad = 0.1
        if (u, v) in seen:
            rad = seen.get((u, v))
            rad = (rad + np.sign(rad) * 0.1) * -1
        seen[(u, v)] = rad
        color = sMap.to_rgba(d["weight"])
        e = FancyArrowPatch(n1.center, n2.center, patchA=n1, patchB=n2,
                            arrowstyle='->',
                            connectionstyle='arc3,rad=%s' % rad,
                            mutation_scale=10, lw=0.2, alpha=1.0, color=color)
        ax.add_patch(e)

    # drawing box
    left = -0.5 * H
    bottom = -len(neurons) * H + 0.5 * H
    width = 1 * H
    height = len(neurons) * H
    box = Rectangle((left, bottom), width, height, fill=False, clip_on=False,
                    lw=0.25)
    ax.add_patch(box)

    # writing text
    for n in neurons:
        ax.text(0.5 * W, -n * H, str(n + 1),
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='k')
    for t in range(maxDelay + 1):
        ax.text(-t * W, -len(neurons) * H - 0.5 * H, str(-t),
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='k')

    # drawing arrow
    ax.arrow(-maxDelay * W, -len(neurons) * H, maxDelay * W - 0.2, 0, width=0.01,
             color="k", clip_on=False, head_width=.5, head_length=1.0)

    ax.autoscale()
    ax.axis("off")
    ax.axis('equal')
    ax.tick_params(axis='both', which='both', bottom='off', top='off',
                   right="off", left="off", labelleft="off", labelbottom='off')
    ax.set_xlim([-W * 9.1, W * 0.1])
    ax.set_ylim([-H * 8, H / 2.])
    print ax.get_xlim()
    print ax.get_ylim()


def draw_color_map():
    x = np.linspace(0, 1, 256)
    x = np.vstack((x, x))

    fig = plt.figure(figsize=(12, 1))
    ax = fig.add_subplot(111)
    ax.imshow(x, aspect='auto', cmap=plt.get_cmap('coolwarm'))
    plt.xlim(min(x[0]), max(x[0]))

    L, B, W, H = list(ax.get_position().bounds)
    font = 14
    fig.text(L, B + H, -100, va='bottom', ha='center', fontsize=font)
    fig.text(L + W * 0.25, B + H, -50, va='bottom', ha='center', fontsize=font)
    fig.text(L + W * 0.5, B + H, 0, va='bottom', ha='center', fontsize=font)
    fig.text(L + W * 0.75, B + H, 50, va='bottom', ha='center', fontsize=font)
    fig.text(L + W, B + H, 100, va='bottom', ha='center', fontsize=font)
    fig.text(L + W * 0.5, B + H * 1.3, "Weight", va='bottom', ha='center',
             fontsize=font)

    plt.tick_params(axis='both', which='both', bottom='on', top='on',
                    right="off", left="off", labelleft="off")
    plt.xticks([0, 64, 128, 192, 256], [-20, -10, 0, 10, 20], fontsize=font)
    plt.xlabel("Bias", fontsize=font)
    # plt.show()

    filename = "heatmap.png"
    fig.savefig(filename, format="png", bbox_inches='tight')

    filename = "heatmap.pdf"
    fig.savefig(filename, format="pdf", bbox_inches='tight')


if __name__ == "__main__":

    # run WriteNetwork.java before running this module

    directory = "../Results/SCIENCE/Single3b/"

    # nlist = [0, 10, 1000, 100000, 120000, 130000]
    files = os.listdir(directory)
    nlist_all = list()
    for f in files:
        if f[-3:] != "csv":
            continue
        n = int(f.split("param")[0][2:])
        nlist_all.append(n)
    nlist_all = sorted(nlist_all)
    print "Found steps:", nlist_all
    nlist = [nlist_all[0]] + nlist_all[-2:]
    for n in nlist_all[1:-2]:
        if str(n)[0] == "1":
            nlist.append(n)
    nlist = sorted(nlist)
    print "Using steps:", nlist

    title_font = {'fontname': 'times', 'size': '10', 'color': 'black',
                  'weight': 'normal', 'verticalalignment': 'bottom'}

    plt.axis('off')
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right="off", left="off", labelleft="off",
                    labelbottom='off')

    for weight_type in ["U", "V"]:

        for i in range(len(nlist)):
            filename = directory + "NN" + str(nlist[i]) + "param3_3delay9.csv"
            bias, delay, U, V = read(filename)
            neurons = bias.keys()

            # node = (time, neuron)
            nodelist = list()
            maxDelay = max(delay.values())
            print "max delay:", maxDelay
            for t in range(maxDelay + 1):
                for n in neurons:
                    node = (t, n)
                    nodelist.append(node)

            # edge = (pre-node, post-node, weight)
            edgelist = list()
            for m in neurons:
                for n in neurons:
                    pre = (delay[(m, n)], m)
                    post = (0, n)
                    if weight_type == "U":
                        weight = sum(U[(m, n)]) / 100.
                    else:
                        weight = sum(V[(m, n)]) / 100.
                    edge = (pre, post, weight)
                    edgelist.append(edge)

            G = nx.DiGraph()
            G.add_nodes_from(nodelist)
            for n in G:
                G.node[n]["bias"] = bias[n[1]] / 20.
                G.node[n]["pos"] = [-n[0], -n[1]]
            G.add_weighted_edges_from(edgelist)

            minifig = plt.figure(figsize=(8, 3.5))  # ,dpi=300)
            miniax = minifig.add_subplot(111)
            drawDyBM(G, miniax)
            filename = "fig/NN_" + weight_type + "" + str(nlist[i]) + ".png"
            minifig.savefig(filename, format="png", bbox_inches='tight')
            filename = "fig/NN_" + weight_type + "" + str(nlist[i]) + ".pdf"
            minifig.savefig(filename, format="pdf", bbox_inches='tight')

    #
    # Color map
    #
    draw_color_map()

    print "Done."

