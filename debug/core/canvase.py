#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ..util.helpers import Logger
from ..util.conf import *
from vispy import gloo, app
from vispy.gloo import VertexBuffer, IndexBuffer
import numpy as np

import math
import threading
import Queue

log = Logger(PRINT_LOG_FILE, V_DEBUG, real_time = True)

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
    vertices = np.repeat(centers, 4, 0) + np.tile(vertices_shift, (n_item, 1)).astype(np.float32)
    textures = np.c_[(np.tile(tex_index, (n_item, 1)), np.repeat(np.linspace(0,1,n_item),4))].astype(np.float32)
    indices = np.tile([0,1,2,1,2,3], n_item) + np.repeat( 4 * np.arange(n_item), 6).astype(np.uint32)
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

    # Compute vertexes
    width, height = 1/float(n_col) * inter_space, 1/float(n_row) * inter_space
    vertices, indices, textures = build_frames(centers, width, height)
    return vertices, indices, textures


image_dim = 28, 28
n_rows = 5.
n_col = 2.
n_item = n_rows * n_col

#base_index = np.asarray([[0,0],[1,0],[0,1],[1,1]])
#base_index = np.tile(base_index, (n_item,1)).astype(np.float32)
#tex_index   = np.copy(base_index)
#tex_helper  = np.repeat(np.linspace(0,1,n_item),4)
#tex_index   = np.c_[ tex_index, tex_helper].astype(np.float32)

positions, indices, texture = create_table(n_item, 2)
indices = IndexBuffer(indices.astype(np.uint32))



VERT_SHADER = """
// vertex shader

attribute vec2 positions;
attribute vec3 a_texindex;

varying vec3 v_texindex;

void main (void) {
    v_texindex = a_texindex;
    gl_Position = vec4(positions.x, positions.y, 0, 1);
}
"""

#### Should find a better texture way
#### This hack is dirty
FRAG_SHADER = """
// texture fragment shader
uniform sampler3D u_texture;
varying vec3 v_texindex;

void main()
{
    //float v_texcoord = v_texindex.x;
    float clr = texture3D(u_texture, v_texindex).r;
    clr = clr - 0.5;
    if ( clr > 0){
        gl_FragColor = vec4( 4*clr, 0., 0., 1.);
    }
    else {
        gl_FragColor = vec4(0., 0., -4* clr, 1.);
    }
}
"""

class Processor(object):
    """
        The processor handles all the Preprocessing of the data before display.
        the controler sends it the model structure and the orders to display.
    """
    def __init__(self, targets):
        #self.targets Should be allways equal to control ones
        self.targets = targets

    def order(self, target):
        """
            This method should set the number of display frames.
            Returns N, the number of frames to be displayed and transmits it
            to Canvas. The canvas deals with the computation of the vertices.

        """
        return True

    def set_model_struct(self, struct):
        self.struct = struct

    def process_simple_lgd(self, data):
        """
        """
        x = data[0].transpose().astype(np.float32)
        x = x.reshape(10, 28, 28)
        log.debug('x shape ={}'.format(x.shape))
        log.debug('max(data)={}'.format(np.max(x)))
        log.debug('min(data)={}'.format(np.min(x)))
        return x

    def process_target(self, target, data):
        """
            target structure is ['solo/cumul', layerName, nodeId]
        """
        # Once we know the kind of order we are able to make,
        # we should find heuristics to compute them
        if target[0] == 0 and target[1] == None and target[2] == -1:
            return self.process_simple_lgd(data)
        else:
            return False

    def process_data(self, data):
        """
            Process data received from the queue.
        """
        ret = []
        for target in self.targets:
            ret.append(self.process_target(target, data))
        ### Should actually concatenate on first dimension, and check that
        ### first dimension is actually equal to 6 * the vertices or n_display_frame
        if len(ret) > 0:
            return ret[0]
        else:
            print 'Nothing returned'


class Canvas(app.Canvas):
    def __init__(self, transmit, processor):
        app.Canvas.__init__(self)
        self.processor = processor

        # Process Thread Control.
        self.running = False
        self.printing = True
        self.transmit = transmit

        # Program definition
        self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        #self._program.bind(vertices)
        self._program['a_texindex'] = texture#tex_index.astype(np.float32)
        self._program['positions'] = positions.astype(np.float32)
        self._program['u_texture'] = np.random.rand(10,*image_dim).astype(np.float32)
        self.show()
        self.start_running()
        self.start_printing()

    def start_printing(self):
        self.printing = True

    def stop_printing(self):
        self.printing = False

    def stop_running(self):
        self.running = False

    def start_running(self):
        if not self.running:
            self.running = True
            thr = threading.Thread(target = self.process_loop, args = ())
            thr.start()

    ####
    ####    Internal Methods
    ####
    def process_loop(self):
        while self.running:
            try:
                data = self.transmit.get(timeout = 0.5)
                x = self.processor.process_data(data)
                if self.printing and x is not None:
                    self._program['u_texture'].set_data(x + 0.5)
                    self.update()
                    log.debug('Has updated')
                else:
                    pass
            except Queue.Empty:
                pass

    def on_draw(self, event):
        gloo.clear()
        self._program.draw('triangles', indices)

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
