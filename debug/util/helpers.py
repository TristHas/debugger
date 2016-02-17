#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, time
from conf import *

class Logger():
    def __init__(self, filename, level = V_INFO, real_time = True):
        self.lev = level
        if not  os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.file = open(filename, 'w')
        self.real_time = real_time

    def hist(self, mess):
        if self.lev >= V_HISTORY:
            message = "[H]{}:{}\n".format(time.time(), mess)
            self.file.write(message)
            if self.real_time:
                self.file.flush()
                os.fsync(self.file)

    def warn(self, mess):
        if self.lev >= V_WARN:
            message = "[W]{}:{}\n".format(time.time(), mess)
            self.file.write(message)
            if self.real_time:
                self.file.flush()
                os.fsync(self.file)

    def info(self, mess):
        if self.lev >= V_INFO:
            message = "[I]{}:{}\n".format(time.time(), mess)
            self.file.write(message)
            if self.real_time:
                self.file.flush()
                os.fsync(self.file)

    def verb(self, mess):
        if self.lev >= V_VERBOSE:
            message = "[V]{}:{}\n".format(time.time(), mess)
            self.file.write(message)
            if self.real_time:
                self.file.flush()
                os.fsync(self.file)

    def debug(self, mess):
        if self.lev >= V_DEBUG:
            message = "[D]{}:{}\n".format(time.time(), mess)
            self.file.write(message)
            if self.real_time:
                self.file.flush()
                os.fsync(self.file)

    def error(self, mess):
        if self.lev >= V_ERROR:
            message = "[E]{}:{}\n".format(time.time(), mess)
            self.file.write(message)
            if self.real_time:
                self.file.flush()
                os.fsync(self.file)


import timeit, time
class WithTimer:
    def __init__(self, title = '', quiet = False):
        self.title = title
        self.quiet = quiet

    def elapsed(self):
        return time.time() - self.wall, time.clock() - self.proc

    def enter(self):
        '''Manually trigger enter'''
        self.__enter__()

    def __enter__(self):
        self.proc = time.clock()
        self.wall = time.time()
        return self

    def __exit__(self, *args):
        if not self.quiet:
            titlestr = (' ' + self.title) if self.title else ''
            print 'Elapsed%s: wall: %.06f, sys: %.06f' % ((titlestr,) + self.elapsed())

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

def list_to_csv(input):
    return CSV_SEP.join(map(str,input))

def send_data(soc, mess):
    soc.sendall(mess)
    while True:
        data = soc.recv(8)
        if data == SYNC:
            break
        if data == FAIL:
            break
    return data

def recv_data(soc):
    data = soc.recv(4096)
    soc.sendall(SYNC)
    return data




