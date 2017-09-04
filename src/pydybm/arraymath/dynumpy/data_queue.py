"""``numpy``-based implementation of data queues
"""

__author__ = "Taro Sekiyama"
__copyright__ = "(C) Copyright IBM Corp. 2016"


class DataQueue:
    def __init__(self, data):
        self._data = data

    def __iter__(self):
        return enumerate(self._data)
