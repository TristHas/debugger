# -*- coding: utf-8 -*-
from debugger.conf import *
import time
import os

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
