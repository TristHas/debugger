#!/usr/bin/env python
# -*- coding: utf-8 -*-
from debugger.logging import Logger
from debugger.conf import *
from vispy import gloo, app
import numpy as np
import threading
import Queue

log = Logger(CANVAS_LOG_FILE, V_DEBUG, real_time = True)

###
###     SHADERS
###

VERT_SHADER = """
// vertex shader

attribute vec2 positions;
attribute vec3 a_texindex;
varying   vec3 v_texindex;

void main (void) {
    v_texindex = a_texindex;
    gl_Position = vec4(positions.x, positions.y, 0, 1);
}
"""

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

class Canvas(app.Canvas):
    def __init__(self, transmit, processor):
        app.Canvas.__init__(self)
        gloo.set_state(clear_color=(0.0, 0.0, 0.0, 1.00))
        self.processor = processor
        # Process Thread Control.
        self.running = False
        self.printing = True
        self.transmit = transmit

        # Program definition
        self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self._program['a_texindex'] = [[[0,0,0]]]
        self._program['positions']  = [[0,0]]
        self._program['u_texture']  = [[np.array([0.,0.]).astype(np.float32)]]
        self.indices                = gloo.IndexBuffer([0,0,0])
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
    def process_panes(self):
        if self.processor.has_updated:
            self.processor.has_updated = False
            self._program['a_texindex'].set_data(self.processor.texindex)
            self._program['positions'].set_data(self.processor.positions)
            self._program['u_texture'].set_data(self.processor.textures + 0.5)
            self.indices = self.processor.index_buffer
            self.update()

    def process_loop(self):
        while self.running:
            try:
                # Synchronisation problem if order is passed and data received while waiting for queue timeout?
                # Should time the process_panes method for performance
                self.process_panes()
                data = self.transmit.get(timeout = 0.5)
                self.processor.process_data(data)
                if self.printing:
                    self._program['u_texture'].set_data(self.processor.textures + 0.5)
                    self.update()
                    log.debug('Has updated')
                else:
                    pass
            except Queue.Empty:
                pass

    def on_draw(self, event):
        gloo.clear()
        if self.processor._check_integrity():
            self._program.draw('triangles', self.indices)
        else:
            log.error('processor integrity check failed before drawing')

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
