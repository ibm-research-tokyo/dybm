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


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def subplot(N, n, i, title=True):
    h = .8 / (2 * N)
    w = 1 / 7.

    cuelen = 5
    runlen = 7

    # forward cue
    ax = plt.axes([0, float(N - i) / N, cuelen * w, h])
    image = np.array(forwardCue).T
    ax.imshow(image, cmap=plt.cm.gray, interpolation="nearest")
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right="off", left="off", labelleft="off",
                    labelbottom='off')

    # forward prediction
    if N == 1:
        ax = plt.axes([3.3 * w, float(N - i) / N, runlen * w, h])
    else:
        ax = plt.axes([2.5 * w, float(N - i) / N, runlen * w, h])
    image = np.array(x[n]).T
    ax.imshow(image, cmap=plt.cm.gray, interpolation="nearest")
    if title:
        if n == 0:
            ax.set_title('Before training', **title_font)
        else:
            if n == 1:
                ax.set_title('After ' + str(n) + " iteration of training",
                             **title_font)
            else:
                num = str(n)
                if len(num) > 3:
                    num = num[:-3] + "," + num[-3:]
                if len(num) > 7:
                    num = num[:-7] + "," + num[-7:]
                ax.set_title('After ' + num + " iterations of training",
                             **title_font)
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right="off", left="off", labelleft="off",
                    labelbottom='off')

    # backward cue
    ax = plt.axes([0, float(2 * N - 2 * i - .9) / (2 * N), cuelen * w, h])
    image = np.array(backwardCue).T
    ax.imshow(image, cmap=plt.cm.gray, interpolation="nearest")
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right="off", left="off", labelleft="off",
                    labelbottom='off')
    if i == N:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plt.arrow(xlim[0], ylim[0] + 1, xlim[1] - xlim[0] - 1.5, 0, width=.1,
                  color="k", clip_on=False, head_width=1., head_length=1.5)

    if N == 1:
        ax = plt.axes(
            [3.3 * w, float(2 * N - 2 * i - .9) / (2 * N), runlen * w, h])
    else:
        ax = plt.axes(
            [2.5 * w, float(2 * N - 2 * i - .9) / (2 * N), runlen * w, h])
    image = np.array(y[n]).T
    ax.imshow(image, cmap=plt.cm.gray, interpolation="nearest")
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right="off", left="off", labelleft="off",
                    labelbottom='off')
    if i == N:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plt.arrow(xlim[0], ylim[0] + 1, xlim[1] - xlim[0] - 1.5, 0, width=.1,
                  color="k", clip_on=False, head_width=1., head_length=1.5)


if __name__ == "__main__":

    from mirror import x, y, forwardCue, backwardCue

    title_font = {'fontname': 'Times', 'size': '16', 'color': 'black',
                  'weight': 'normal', 'verticalalignment': 'bottom'}

    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    right="off", left="off", labelleft="off",
                    labelbottom='off')

    nlist = sorted(x.keys())

    for i in range(len(nlist)):
        n = nlist[i]
        fig = plt.figure(figsize=(6, 2))
        subplot(1, n, 1, False)
        filename = "fig/MirrorSCIENCE" + str(n) + ".png"
        fig.savefig(filename, format="png", bbox_inches='tight')
        filename = "fig/MirrorSCIENCE" + str(n) + ".pdf"
        fig.savefig(filename, format="pdf", bbox_inches='tight')

    print "Done."

