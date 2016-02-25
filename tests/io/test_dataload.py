#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from debugger.core.canvas import *
from debugger.io.dataload import load_mnist, SetManager, init_dataset

def test_load_mnist():
    sets = train, valid, test  = load_mnist()
    for set in test, valid:
        result = init_dataset(set)
        assert len(result) == 4
        assert result[3].shape == (10000,)
        assert result[2].shape == (10000,)
        assert sum(result[2]) == 0

    result = init_dataset(train)
    assert len(result) == 4
    assert result[3].shape == (50000,)
    assert result[2].shape == (50000,)
    assert sum(result[2]) == 0

def test_SetManager_init(mnist_sets):
    sets = {}
    sets[]
    sm = SetManager(mnist_sets)




