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

"""``cupy``-based implementation of FIFO queues
"""

__author__ = "Taro Sekiyama"


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
