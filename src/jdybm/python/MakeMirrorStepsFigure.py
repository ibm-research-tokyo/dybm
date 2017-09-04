# -*- coding: utf-8 -*-

__author__ = "Takayuki Osogami"
__copyright__ = "(C) Copyright IBM Corp. 2015"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    directory = "../Results/SCIENCE/Mirror3b/"

    from mirror import x
    nlist = sorted(x.keys())
    last = nlist[-1]
    # last = 1785845

    filename = directory + "Backward" + str(last) + "_cue5param3_3delay9.csv"

    f = open(filename)
    backward = list()
    for line in f:
        backward.append(int(line.strip()))
    f.close()

    filename = directory + "Forward" + str(last) + "_cue5param3_3delay9.csv"

    f = open(filename)
    forward = list()
    for line in f:
        forward.append(int(line.strip()))
    f.close()

    fig = plt.figure()
    ax = plt.gca()
    n = len(forward)
    ax.plot(2 * np.arange(n) + 1, forward, marker='.', linewidth=0.1, c='red',
            alpha=1, markeredgecolor='none')
    ax.plot(2 * np.arange(n) + 2, backward, marker='.', linewidth=0.1, c='blue',
            alpha=1, markeredgecolor='none')
    ax.plot([], [], '.', c='red', alpha=1, markeredgecolor='none',
            label="forward")
    ax.plot([], [], '.', c='blue', alpha=1, markeredgecolor='none',
            label="backward")
    plt.legend()
    ymax = max(forward + backward)
    plt.xlim([0.99, 2 * n * 1.01])
    plt.ylim([0, ymax * 1.01])
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of steps in the last iteration")
    ax.set_xscale('log')

    filename = "StepsSCIENCE.png"
    fig.savefig("fig/" + filename, format="png", bbox_inches='tight')

    filename = "StepsSCIENCE.pdf"
    fig.savefig("fig/" + filename, format="pdf", bbox_inches='tight')

    print "Done."

