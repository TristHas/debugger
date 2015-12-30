##!/usr/bin/env python
## -*- coding: utf-8 -*-

from debugger.debug.core.canvase import Canvas
from debugger.debug.util.mnist_loader import load_data
from debugger.debug.util.helpers import Logger
from debugger.debug.util.conf import *
from debugger.models.models import LogisticModel
from debugger.models.trainer import NLL_Trainer
from debugger.network.easy_client import LightClient

import threading, Queue, sys

from vispy import app, use
use(app = 'PyQt5')
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

log = Logger(SGD_LOG_FILE, level = V_DEBUG)

###
###     Easy_client
###
class ControlWindow(QWidget):
    def __init__(self, transmit, parent = None, ip = None):
        QWidget.__init__(self, parent)
        self.lineEdit = QLineEdit("")

        formLayout = QFormLayout()
        formLayout.addRow(self.tr("&Command:"), self.lineEdit)
        layout = QGridLayout()
        layout.addLayout(formLayout, 1, 0, 1, 3)
        self.setLayout(layout)
        self.setWindowTitle(self.tr("Control Prompt"))

        if ip is None:
            self.lineEdit.returnPressed.connect(self.process_order_local)
            self.init_local(transmit)
        else:
            self.init_remote(ip, transmit)
            self.lineEdit.returnPressed.connect(self.process_order_remote)

        self.show()

    def init_local(self, transmit):
        datasets = load_data('mnist.pkl.gz')
        train_set = datasets[0]
        valid_set = datasets[1]
        test_set = datasets[2]
        model = LogisticModel(input_shape=(28, 28), n_out=10)
        self.trainer = NLL_Trainer(transmit, model, train_set, valid_set, test_set)
        self.trainer.launch_training()

    def init_remote(self, ip, transmit):
        self.client = LightClient(ip, transmit)

    def process_order_remote(self):
        text = unicode(self.lineEdit.text())
        if text == 'start':
            self.client.start_training()
            self.lineEdit.setText('')
        if text == 'pause':
            self.client.pause_training()
            self.lineEdit.setText('')
        if text == 'resume':
            self.client.resume_training()
            self.lineEdit.setText('')
        if text == 'stop':
            self.client.stop_training()
            self.lineEdit.setText('')
        if text == 'record':
            self.client.start_record('full_layer', 'l_1')
            self.client.start_receive()
            self.lineEdit.setText('')

    def process_order_local(self):
        text = unicode(self.lineEdit.text())
        if text == 'start':
            self.trainer.resume_training()
            self.lineEdit.setText('')
        if text == 'pause':
            self.trainer.pause_training()
            self.lineEdit.setText('')
        if text == 'record':
            self.trainer.add_target('l_1')
            self.trainer.start_record()
            self.lineEdit.setText('')


def main():
    x = app.Application()
    transmit = Queue.Queue()
    c = Canvas(transmit)
    w = ControlWindow(transmit)
    qt_app = x.native
    sys.exit(qt_app.exec_())

if __name__ == '__main__':
    main()
