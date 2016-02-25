#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import math


###
###     Sets
###

def to_list_of_sets(data):
    if isinstance(data, dict):
        return [set(data[key]) for key in data]
    elif isinstance(data, list):
        return [set(item) for item in data]
    else:
        raise TypeError('to_list_of_sets function received argument {} of type {}'.format(data, type(data)))

def no_overlap(data):
    sets = to_list_of_sets(data)
    return len(set.union(*sets)) == sum(len(item) for item in sets)

def is_contiguous_set(data):
    if no_overlap(data):
        if isinstance(data, dict):
            data = [data[key] for key in data]
        elif isinstance(data, list):
            # not great
            if type(data[0]) == list:
                data = list(itertools.chain(*data))
            data.sort()
            should_be    = range(np.min(data), np.max(data) + 1)
            return data == should_be
        else:
            raise TypeError('is_contiguous_set function received argument {} of type {}'.format(data, type(data)))

def union(data):
    sets = to_list_of_sets(data)
    return set.union(*sets)
