#!/usr/bin/env python
# -*- coding: utf-8 -*-
 import numpy as np
 from debugger.core.canvase import *
 import random


@pytest.mark.parametrize("n_row, n_col", zip(random_int_list(), random_int_list()))
def test_method_create_table_centers_ii(n_col, n_row):
    """
        DOC
    """
    centers = create_table_centers(n_col, n_row)
    assert centers.shape == (2, n_row * n_col)
    assert centers.min > 0 and centers.max < 1
    assert centers / (1/float(n_col), 1/float(n_row))
    #assert they are all different

@pytest.mark.parametrize("n_row, n_col, inter_space", zip(random_int_list(), random_int_list(), random_float_list()))
def test_build_frames_ii(n_col, n_row, inter_space):
    centers = create_table_centers(n_col, n_row)
    build_frames(centers, n_row, n_col)
    assert True
