#!/usr/bin/env python
# -*- coding: utf-8 -*-
#import debugger.tests.tools
from debugger.core.canvas import *
from tools.number_tools import *
import pytest
import numpy as np
import random

@pytest.mark.parametrize("n_item, n_row", zip(random_int_list(), random_int_list()))
def create_col_row():
    """
        DOC
    """
    n_col, n_row = create_col_row(n_tiem, n_row)
    assert n_col * n_row >= n_item

@pytest.mark.parametrize("n_row, n_col", zip(random_int_list(), random_int_list()))
def test_method_create_table_centers_ii(n_col, n_row):
    """
        DOC
    """
    centers = create_table_centers(n_col, n_row)
    assert centers.shape == (n_row * n_col, 2)
    assert centers.min() > -1 and centers.max() < 1
    assert isClose(0.5, (centers * (n_col, n_row) % 1))
    dist = minimum_dist(centers)
    assert equalOrInf(1/float(n_col), dist)  or equalOrInf(1/float(n_row), dist)

@pytest.mark.parametrize("width, height", zip(random_int_list(), random_int_list()))
def test_build_frames_aii(width, height):
    """
        DOC
    """
    centers = create_table_centers(4, 4)
    n_items = centers.shape[0]
    vertices, indices, textures = build_frames(centers, width, height)
    for i in range(n_items):
        v_base_index = 4 * i
        assert (vertices[v_base_index] - vertices[v_base_index + 1] == [-width, 0]).all()
        assert (vertices[v_base_index] - vertices[v_base_index + 2] == [0, -height]).all()
        assert (vertices[v_base_index + 3] - vertices[v_base_index + 1] == [0, height]).all()
        assert (vertices[v_base_index + 3] - vertices[v_base_index + 2] == [width, 0]).all()
        t_base_index = 4 * i
        i_base_index = 6 * i

@pytest.mark.parametrize("n_item, n_row , inter_space", [[10, 2, 0.9]])#zip(random_int_list(), random_int_list(), random_float_list()))
def test_create_table_iif(n_item, n_row , inter_space):
    v, i, t = create_table(n_item, n_row , inter_space)
    # Return shapes
    assert v.shape == (4 * n_item, 2)
    assert t.shape == (4 * n_item, 4)
    assert i.shape == (6 * n_item, )
    # Window integration
    assert v.min() > -1 and v.max() < 1




