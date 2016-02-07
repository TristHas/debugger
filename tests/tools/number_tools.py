#!/usr/bin/env python
# -*- coding: utf-8 -*-

import np
import random
import math

def random_int_list(x = 10, min = 0, max = 100):
    """
        Returns a list of x random int between min and max.
    """
    return [random.randint(min, max) for i in range(x)]

def random_float_list(x = 10, min = 0, max = 1):
    """
        Returns a list of x random int between min and max.
    """
    return [random.randint(min, max) for i in range(x)]
