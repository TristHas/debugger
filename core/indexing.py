#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math


###
###     GL vertices
###
def create_table_centers(n_col, n_row):
    """
        DOC
    """
    y_coords = -1 + 2/float(n_row) * (np.arange(n_row) + 0.5)
    x_coords = -1 + 2/float(n_col) * (np.arange(n_col) + 0.5)
    return np.transpose([np.tile(x_coords, len(y_coords)), np.repeat(y_coords, len(x_coords))])

def build_frames(centers, width, height ):
    """
        DOC
    """
    n_item = centers.shape[0]
    dim = np.array([width, height])
    tex_index = np.array([  [0, 1],
                            [1, 1],
                            [0, 0],
                            [1, 0],
                        ])
    vertices_shift = np.array([   [- width, - height],
                                        [width, -height],
                                        [-width, height],
                                        [width, height]
                                    ])
    vertices = (np.repeat(centers, 4, 0) + np.tile(vertices_shift, (n_item, 1))).astype(np.float32)
    textures = np.c_[(np.tile(tex_index, (n_item, 1)), np.repeat(np.linspace(0,1,n_item),4))].astype(np.float32)
    indices = (np.tile([0,1,2,1,2,3], n_item) + np.repeat( 4 * np.arange(n_item), 6)).astype(np.uint32)
    return vertices, indices, textures

def create_col_row(n_item, n_row = None):
    if n_row is None:
        n_col = float(math.ceil(math.sqrt(n_item)))
        n_row = math.ceil(n_item / n_col)
    else:
        n_row = float(n_row)
        n_col = float(math.ceil(float(n_item) / n_row))
    return n_col, n_row

def create_table(n_item, n_row = None, inter_space = 0.9):
    """
        DOC
    """
    # Compute rows/columns
    n_col, n_row = create_col_row(n_item, n_row)
    # Compute centers
    centers = create_table_centers(n_col, n_row)
    centers = centers[:n_item, :]
    # Compute vertices
    width, height = 1/float(n_col) * inter_space, 1/float(n_row) * inter_space
    vertices, indices, textures = build_frames(centers, width, height)
    return vertices, indices, textures

def reallocate_table(table, partition, center):
    scale = 1. / partition
    new_table = table * scale
    shift = [-1, -1] + scale
    reallocated_table = shift + 2 * center * scale + new_table
    return reallocated_table


###
###     GL Indices
###
def shift_indices(indices, n_old_item):
     indices_shift = np.ones(indices.shape) * n_old_item
     return indices_shift + indices

def adjust_texindex(texindex):
     ### TODO
     n_item         = texindex.shape[0] / 4
     indices        = np.repeat(np.linspace(0,1,n_item),4)
     texindex[:,2]  = indices

def resize_nearest(data, size):
    x_indices = np.arange(data.shape[0])
    y_indices = np.linspace(0, data.shape[1], size[0]).astype(int).clip(0, data.shape[1] - 1)
    z_indices = np.linspace(0, data.shape[2], size[1]).astype(int).clip(0, data.shape[2] - 1)
    return data[np.meshgrid(x_indices, y_indices, z_indices)].swapaxes(0, 1)


