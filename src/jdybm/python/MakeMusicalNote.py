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
from matplotlib.patches import Circle, Ellipse


if __name__ == "__main__":

    from musician import x

    nlist_all = sorted(x.keys())
    print "Found steps:", nlist_all
    nlist = [nlist_all[0]] + nlist_all[-2:]
    for n in nlist_all[1:-2]:
        if str(n)[0] == "1":
            nlist.append(n)
    nlist = sorted(nlist)
    print "Using steps:", nlist

    for N in nlist:
        print "drawing", N

        offset = -5
        space = 6
        mid = 3
        w = [0, (len(x[N]) + 1) * space + 14 * mid]

        width = (len(x[N]) + 1) * space
        height = 12 * 2 + offset

        fig = plt.figure(figsize=(width / 2., height / 2.))  # ,dpi=300)
        plt.axis('off')
        plt.tick_params(axis='both', which='both', bottom='off', top='off',
                        right="off", left="off", labelleft="off",
                        labelbottom='off')
        ax = fig.add_subplot(111)

        # drawing five lines
        for n in [1, 2, 3, 4, 5]:
            ax.plot(w, [2 * n, 2 * n], color="black")
            ax.plot(w, [offset - 2 * n, offset - 2 * n], color="black")

        # drawing onpu
        nmid = 0
        for i in range(len(x[N])):
            if i % 4 == 1:
                nmid += 1
                ax.plot([space * i + nmid * mid + 1] * 2,
                        [2, 10],
                        color="black")
                ax.plot([space * i + nmid * mid + 1] * 2,
                        [offset - 2, offset - 10],
                        color="black")
            pattern = x[N][i]
            if max(pattern[2:7]) == 1:
                # right bar
                for j in range(7):
                    if pattern[j] == 1:
                        row = 7 - j
                        ax.plot([space * (i + 1) + nmid * mid + 1.2] * 2, [row, row + 7],
                                color="black")
                        pre = 0
                        for k in range(j + 1, 7):
                            if pattern[k] == 1:
                                pre += 1
                            else:
                                break
                        if pre % 2 == 1:
                            c = Ellipse([space * (i + 1) + nmid * mid + 2.4, row],
                                        width=2.5, height=1.8, angle=30,
                                        color="black")
                        else:
                            c = Ellipse([space * (i + 1) + nmid * mid, row], width=2.5,
                                        height=1.8, angle=30, color="black")
                        ax.add_patch(c)
            elif max(pattern[:2]) == 1:
                # left bar
                for j in range(7):
                    if pattern[j] == 1:
                        row = 7 - j
                        ax.plot([space * (i + 1) + nmid * mid - 1.2] * 2, [row - 7, row],
                                color="black")
                        pre = 0
                        for k in range(j):
                            if pattern[k] == 1:
                                pre += 1
                            else:
                                pre = 0
                        if pre % 2 == 1:
                            c = Ellipse([space * (i + 1) + nmid * mid - 2.4, row],
                                        width=2.5, height=1.8, angle=30,
                                        color="black")
                        else:
                            c = Ellipse([space * (i + 1) + nmid * mid, row], width=2.5,
                                        height=1.8, angle=30, color="black")
                        ax.add_patch(c)

            for j in range(7, len(pattern)):
                if pattern[j] == 1:
                    row = 7 - j + offset
                    ax.plot([space * (i + 1) + nmid * mid - 1.2] * 2, [row - 7, row],
                            color="black")
                    pre = 0
                    for k in range(7, j):
                        if pattern[k] == 1:
                            pre += 1
                        else:
                            pre = 0
                    if pre % 2 == 1:
                        c = Ellipse([space * (i + 1) + nmid * mid - 2.4, row], width=2.5,
                                    height=1.8, angle=30, color="black")
                    else:
                        c = Ellipse([space * (i + 1) + nmid * mid, row], width=2.5,
                                    height=1.8, angle=30, color="black")
                    ax.add_patch(c)
                    if j == 7:
                        ax.plot([space * (i + 1) + nmid * mid - 2, space * (i + 1) + nmid * mid + 2],
                                [offset, offset], color="black")

        ax.set_xlim(w)
        ax.set_ylim(offset - 12, 12)

        # plt.show()
        # break

        filename = "fig/music" + str(N) + ".png"
        fig.savefig(filename, format="png", bbox_inches='tight')

        filename = "fig/music" + str(N) + ".pdf"
        fig.savefig(filename, format="pdf", bbox_inches='tight')

    print "Done."
