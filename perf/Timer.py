# -*- coding: utf-8 -*-

import timeit, time

class Timer:
    def __init__(self):
        self.acc = {}
        self.rep = {}

    def time(self, func, *args):
        start = timeit.default_timer()
        result = func(*args)
        end = timeit.default_timer()
        if func not in self.acc:
            self.acc[func] = 0
            self.rep[func] = 0
        self.acc[func] += end - start
        self.rep[func] += 1
        return result

    def get_avg_time(self, func):
        if func not in self.acc:
            return None
        else:
            return self.acc[func] / self.rep[func]

    def get_total_time(self, func):
        if func not in self.acc:
            return None
        else:
            return self.acc[func]
