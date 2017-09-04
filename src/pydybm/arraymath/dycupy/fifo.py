"""``cupy``-based implementation of FIFO queues
"""

__author__ = "Taro Sekiyama"
__copyright__ = "(C) Copyright IBM Corp. 2016"


import cupy


class FIFO:
    def __init__(self, shape):
        self._fifo = cupy.zeros(shape)

    def __len__(self):
        return len(self._fifo)

    def push(self, a):
        b = self._fifo[-1, :]
        self._fifo = cupy.roll(self._fifo, 1, axis=0)
        self._fifo[0, :] = a
        return b

    def to_array(self):
        return self._fifo
