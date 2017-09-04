"""``numpy``-based implementation of FIFO queues
"""

__author__ = "Taro Sekiyama"
__copyright__ = "(C) Copyright IBM Corp. 2016"


import numpy
import collections


class FIFO:
    def __init__(self, shape):
        self._fifo = collections.deque(numpy.zeros(shape))
        self._arr = numpy.array(self._fifo)

    def __len__(self):
        return len(self._fifo)

    def push(self, a):
        b = self._fifo.pop()
        self._fifo.appendleft(a)
        self._arr = numpy.array(self._fifo)
        return b

    def to_array(self):
        return self._arr
