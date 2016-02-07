#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ..util.helpers import Logger
from ..util.conf import *
from vispy import gloo, app
import numpy as np

import math
import threading
import Queue

log = Logger(PRINT_LOG_FILE, V_DEBUG, real_time = True)

image_dim = 28, 28
n_rows = 5.
n_col = 2.
m = n_rows * n_col

base_index = np.asarray([[0,0],[1,0],[0,1],[1,0],[0,1],[1,1]])
base_index = np.tile(base_index, (m,1)).astype(np.float32)
log.debug('base_index')
log.debug(base_index.shape)
log.debug(base_index)

tex_index   = np.copy(base_index)
tex_helper  = np.repeat(np.linspace(0,1,m),6)
tex_index   = np.c_[ tex_index, tex_helper].astype(np.float32)
log.debug("tex_index")
log.debug(tex_index.shape)
log.debug(tex_index)

table_index_x = np.repeat(np.repeat(np.arange(n_rows),6), n_col)
table_index_y = np.tile(np.repeat(np.arange(n_col),6), n_rows)
table_index   = np.c_[table_index_x,table_index_y].astype(np.float32)
log.debug("table_index")
log.debug(table_index.shape)
log.debug(table_index)

# Not used, but could be used with only one index vriable as in imshow example
final_index = np.c_[table_index, base_index]

VERT_SHADER = """
// vertex shader

uniform vec2 u_tdim;
attribute vec2 a_basindex;
attribute vec2 a_tabindex;
attribute vec3 a_texindex;

varying vec3 v_texindex;

void main (void) {
    // Passe tex coords
    v_texindex = a_texindex;
    // Calculate position
    vec2 center = vec2( - 1 + 2* ((a_tabindex.x + 0.5) * u_tdim.x), - 1 + 2* ( (a_tabindex.y + 0.5) * u_tdim.y));
    gl_Position = vec4(center.x + 0.9 * (-1 + 2* a_basindex.x) * u_tdim.x, center.y + 0.9 * (-1 + 2* a_basindex.y) * u_tdim.y, 0.0, 1.0);
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
        #image_dim = 28, 28
        #n_rows = 5.
        #n_col = 2.
        #m = n_rows * n_col
        ## Index pour définir les cadres
        ## Définit consécutivement deux triangles opposés par l'hypoténuse
        #base_index = np.asarray([[0,0],[1,0],[0,1],[1,0],[0,1],[1,1]])
        ## Répète ca le nombre de fois qu'il y a de cadre dans lesquels dessiner
        ## vérifier ses dimensions
        #base_index = np.tile(base_index, (m,1)).astype(np.float32)
        #log.error('Dimensions de base_index: {}'.format(base_index.shape))
        #
        ## Index pour les textures
        #tex_index   = np.copy(base_index)
        #tex_helper  = np.repeat(np.linspace(0,1,m),6)
        #log.error('Dimensions de tex_helper: {}'.format(tex_helper.shape))
        #tex_index   = np.c_[ tex_index, tex_helper].astype(np.float32)
        #log.error('Dimensions de tex_index: {}'.format(tex_index.shape))
        #
        #table_index_x = np.repeat(np.repeat(np.arange(n_rows),6), n_col)
        #table_index_y = np.tile(np.repeat(np.arange(n_col),6), n_rows)
        #table_index   = np.c_[table_index_x,table_index_y].astype(np.float32)
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
        return ret[0]



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
        self._program['a_texindex'] = tex_index
        self._program['a_basindex'] = base_index
        self._program['a_tabindex'] = table_index
        self._program['u_tdim'] = np.asarray([1/n_rows, 1/n_col], dtype = np.float32)
        self._program['u_texture'] = texture = np.random.rand(10,*image_dim).astype(np.float32)
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
                if self.printing:
                    self._program['u_texture'].set_data(x + 0.5)
                    self.update()
                    log.debug('Has updated')
                else:
                    pass
            except Queue.Empty:
                pass

    def on_draw(self, event):
        gloo.clear()#color=(0.0, 0.0, 0.0, 1.0))
        self._program.draw('triangles')

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

def create_table_centers(n_col, n_row):
    """
        DOC
    """
    y_coords = 1/float(n_row) * (np.arange(n_row) + 0.5)
    x_coords = 1/float(n_col) * (np.arange(n_col) + 0.5)
    return np.transpose([np.tile(x_coords, len(y_coords)), np.repeat(y_coords, len(x_coords))])

def build_frames(centers, width, height ):
    """
    DOC
    """
    n_item = centers.shape[0]
    dim = np.array([width, height])
    tex_index = np.array([  [0, 0],
                            [1, 0],
                            [0, 1],
                            [1, 1],
                        ])
    vertices_shift = 0.5 * np.array([   [- width, - height],
                                        [width, -height],
                                        [-width, height],
                                        [width, height],
                                    ])
    vertices = np.repeat(centers, 4, 0) + np.tile(vertices_shift, (n_item, 1))
    tex_ind_temp = np.tile(tex_index, (n_item, 1))
    print tex_ind_temp.shape
    print vertices.shape
    textures =  np.concatenate((vertices, tex_ind_temp), axis = 1)
    indices = np.tile([0,1,2,1,2,3], n_item) + np.repeat( 4 * np.arange(n_item), 6)
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
    centers = create_table_centers(n_row, n_col)
    centers = centers[:n_item, :]

    # Compute vertexes
    width, height = 1/float(n_col) * inter_space, 1/float(n_row) * inter_space
    vertices, indices, textures = build_frames(centers, width, height)
    return vertices, indices, textures
