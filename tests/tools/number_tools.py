#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import math
from scipy.spatial.distance import pdist
import pytest


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

def unique_points(data):
    new_array = [tuple(row) for row in array]
    uniques = np.unique(new_array)
    return len(uniques) == data.shape[0]

def minimum_dist(data):
    return np.min(pdist(data))

def isClose(a, b, tolerance=1e-09):
    return abs((a-b).max()) <= tolerance

def equalOrInf(a, b, tolerance=1e-09):
    return abs((a-b).max()) <= tolerance or a < b

