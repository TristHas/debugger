#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ..util.helpers import Logger
from ..util.conf import *
from vispy import gloo, app
import numpy as np

log = Logger(PRINT_LOG_FILE, V_DEBUG, real_time = True)

image_dim = 28,28
n_rows = 5.
n_col = 2.
m = n_rows * n_col

base_index = np.asarray([[0,0],[1,0],[0,1],[1,0],[0,1],[1,1]])
base_index = np.tile(base_index, (m,1)).astype(np.float32)

tex_index   = np.copy(base_index)
tex_helper  = np.repeat(np.linspace(0,1,m),6)
tex_index   = np.c_[ tex_index, tex_helper].astype(np.float32)

table_index_x = np.repeat(np.repeat(np.arange(n_rows),6), n_col)
table_index_y = np.tile(np.repeat(np.arange(n_col),6), n_rows)
table_index   = np.c_[table_index_x,table_index_y].astype(np.float32)

# Not used, but could be used with only one index vriable as in imshow example
final_index = np.c_[table_index, base_index]

VERT_SHADER = """ // vertex shader

uniform vec2 u_tdim;

attribute vec2 a_basindex;
attribute vec2 a_tabindex;
attribute vec3 a_texindex;

varying vec3 v_texindex;

void main (void) {
    // Pass tex coords
    v_texindex = a_texindex;
    // Calculate position
    vec2 center = vec2( - 1 + 2* ((a_tabindex.x + 0.5) * u_tdim.x), - 1 + 2* ( (a_tabindex.y + 0.5) * u_tdim.y));
    gl_Position = vec4(center.x + 0.9 * (-1 + 2* a_basindex.x) * u_tdim.x, center.y + 0.9 * (-1 + 2* a_basindex.y) * u_tdim.y, 0.0, 1.0);
}
"""

#### Should find a better texture way
#### This hack is dirty
FRAG_SHADER = """ // texture fragment shader
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
    //gl_FragColor.a = 1.0;
}
"""

class Canvas(app.Canvas):
    def __init__(self, transmit):
        app.Canvas.__init__(self)
        # Theano synchronisation
        self.running = False
        self.printing = False
        self.transmit = transmit
        # Program definition
        self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self._program['a_texindex'] = tex_index
        self._program['a_basindex'] = base_index
        self._program['a_tabindex'] = table_index
        self._program['u_tdim'] = np.asarray([1/n_rows, 1/n_col], dtype = np.float32)
        self._program['u_texture'] = np.random.rand(10,*image_dim).astype(np.float32)
        self.show()

    def start_print(self):
        self.printing = True

    def stop_printing(self):
        self.printing = False

    def stop_running(self):
        self.running = False

    def start_running(self):
        if not self.running:
            thr = threading.Thread(target = self.process_loop, args = ())
            thr.start()
            self.running = True

    ####
    ####    Internal Methods
    ####
    def process_loop(self):
        while self.running:
            try:
                data = self.transmit.get(timeout = 1)
                for layer in data:
                    tmp = [data[layer][key] for key in data[layer]]
                    x = np.c_[tmp].astype(np.float32)
                log.debug('max(data)={}'.format(np.max(x)))
                log.debug('min(data)={}'.format(np.min(x)))
                if self.printing:
                    self._program['u_texture'].set_data(x + 0.5)
                    self.update()
                else:
                    pass
            except Queue.Empty:
                pass

    def on_draw(self, event):
        gloo.clear()
        self._program.draw('triangles')

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
