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

"""``cupy``-based implementation of data queues
"""

__author__ = "Taro Sekiyama"


import collections
import numpy as np
import cupy
import chainer


class Iterator:
    def __init__(self, data, prefetch):
        self._idx = 0
        self._queue = collections.deque()
        self._data = data
        self._stream = cupy.cuda.Stream()
        self._prefetch = min(prefetch, len(self._data))

        if self._prefetch <= 0:
            raise ValueError

        for _ in range(self._prefetch):
            e = cupy.cuda.Event(block=True)
            a = cupy.empty_like(self._current_data)
            a.set(np.array(self._current_data, copy=False), self._stream)
            e.record(self._stream)
            self._queue.append((e, a))
            self._idx += 1

        if len(self._queue) > 0:
            self._next = (cupy.cuda.Event(block=True),
                          cupy.empty_like(list(self._queue[0])[1]))

    def next(self):
        if len(self._queue) == 0:
            raise StopIteration

        e, a = self._queue.popleft()

        if self._idx < len(self._data):
            next_e, next_a = self._next
            next_a.set(np.array(self._current_data, copy=False), self._stream)
            next_e.record(self._stream)
            self._queue.append(self._next)
            self._next = (e, a)

        self._idx += 1
        e.synchronize()
        return (self._idx - self._prefetch - 1, a)

    @property
    def _current_data(self):
        return self._data[self._idx]


class DataQueue:
    def __init__(self, data, prefetch=5):
        self._data = data
        self._prefetch = prefetch

    def __iter__(self):
        return Iterator(self._data, self._prefetch)
