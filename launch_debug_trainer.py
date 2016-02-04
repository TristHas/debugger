##!/usr/bin/env python
## -*- coding: utf-8 -*-

import Queue, sys
import debugger
from debugger.debug.core.canvase import Canvas, Processor
from debugger.debug.core.control_window import ControlWindow
from vispy import app, use
use(app = 'PyQt5')

###
###     Easy_client
###

def main():
    x = app.Application()
    transmit = Queue.Queue()
    targets = {}
    # refactor this mess
    # Control needs a reference to p for passing new processing
    # and to pass model structure to p
    # Control needs a reference to target for
    # intelligent add_target() and remove_target() calls to client
    # process needs to share this target reference with control windows
    # for synchronized processing
    # Canvas obviously need processor to handle its computation
    p = Processor(targets)
    c = Canvas(transmit, p)
    w = ControlWindow(transmit, targets, p)
    qt_app = x.native
    sys.exit(qt_app.exec_())

if __name__ == '__main__':
    main()
