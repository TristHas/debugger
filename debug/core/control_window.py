#!/usr/bin/env python
# -*- coding: utf-8 -*-
from debugger.debug.util.mnist_loader import load_data
from debugger.debug.util.helpers import Logger
from debugger.debug.util.conf import *
from debugger.models.models import LogisticModel
from debugger.models.trainer import NLL_Trainer
from debugger.network.client import LightClient, LocalClient

import threading
from sets import Set

from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class ControlWindow(QWidget):
    def __init__(self, transmit, targets, processor, parent = None, ip = None):
        QWidget.__init__(self, parent)
        self.log = Logger(CTRL_LOG_FILE, level = V_DEBUG)
        self.targets = targets
        self.processor = processor
        self.lineEdit = QLineEdit("")
        formLayout = QFormLayout()
        formLayout.addRow(self.tr("&Command:"), self.lineEdit)
        layout = QGridLayout()
        layout.addLayout(formLayout, 1, 0, 1, 3)
        self.setLayout(layout)
        self.setWindowTitle(self.tr("Control Prompt"))
        if ip is None:
            self.init_local(transmit)
        else:
            self.init_remote(ip, transmit)
        self.lineEdit.returnPressed.connect(self.process_command)
        self.show()

    def init_local(self, transmit):
        """
            Instantiate client and struct.
        """
        self.log.info('Init Local')
        train_set, valid_set, test_set = load_data('mnist.pkl.gz')
        model = LogisticModel(input_shape=(28, 28), n_out=10)
        trainer = NLL_Trainer(transmit, model, train_set, valid_set, test_set)
        self.client = LocalClient(trainer)
        self.processor.set_model_struct(model.struct)

    def init_remote(self, ip, transmit):
        #### Deploy data first?
        self.log.info('Init Remote')
        self.client = LightClient(ip, transmit)

    def process_command(self):
        '''
        '''
        text = unicode(self.lineEdit.text())
        if text == 'start':
            self.log.info('Received command {}'.format(text))
            self.client.start_training()
            self.lineEdit.setText('')
        if text == 'pause':
            self.log.info('Received command {}'.format(text))
            self.client.pause_training()
            self.lineEdit.setText('')
        if text == 'resume':
            self.log.info('Received command {}'.format(text))
            self.client.resume_training()
            self.lineEdit.setText('')
        if text == 'stop':
            self.log.info('Received command {}'.format(text))
            self.client.stop_training()
            self.lineEdit.setText('')
        if text.startswith('show '):
            self.log.info('Received command {}'.format(text))
            self.process_show(text)
            self.lineEdit.setText('')
        if text.startswith('set '):
            self.log.info('Received command {}'.format(text))
            self.process_set(text)
            self.lineEdit.setText('')
        if text.startswith('get '):
            self.log.info('Received command {}'.format(text))
            self.process_get(text)
            self.lineEdit.setText('')
        if text.startswith('load'):
            self.log.info('Received command {}'.format(text))
            self.process_load(text)
            self.lineEdit.setText('')


    def process_show(self, text):
        '''
            Process show commands.
            Correct syntax are: show [cumul] layerName [nodeId]
            Displays the requested parameters and returns True if correct syntax
            and correct request (incorrect layerName/nodeId)
            Returns False if incorrect syntax and/or incorrect layerName/nodeId
        '''
        def parse_show(text):
            args = text.split()
            self.log.debug('text={}. args={}'.format(text,args))
            try:
                if args[1] == 'cumul':
                    args.remove('cumul')
                    target  = ['cumul', int(args[1])]
                    layers  = range(target[1])
                else:
                    target  = ['solo', int(args[1])]
                    layers  = [target[1]]
            except (ValueError, IndexError) as e:
                self.log.error('incorrect syntax for {}. Got error: {}'.format(text, e))
                return False, None
            try:
                target.append(int(args[2]))
                target = tuple(target)
            except (ValueError, IndexError) as e:
                target.append(-1)
                target = tuple(target)
            finally:
                return target, layers

        target, layers = parse_show(text)
        self.log.debug(layers)
        if target :
            if not target in self.targets:
                add_target = all(self.client.add_target(x) for x in layers)
                if add_target:
                    order_success = self.processor.order(target)
                    if order_success:
                        self.targets[target] = layers
                        self.client.start_record()
                        self.log.info('Successfully added target {}'.format(target))
                    else:
                        ##TODO: Should remove targets from client
                        self.log.error("Order failed")
                else:
                    ##TODO: Should remove targets from client
                    self.log.error('Client Target Add Failed for  {}'.format(target))
            else:
                self.log.warn('Target {} asked whereas allready in target control list'.format(target))
        else:
            self.log.error('Target not correctly processed'.format(target))

    def process_set(self, text):
        '''
            Process all set commands.
            Correct syntax are: set parameter value
            Returns True if correct syntax and correct request
            Returns False if incorrect syntax and/or request
        '''
        arguments = text.split()
        if len(arguments) != 3:
            self.log.warn('set incorrect syntax')
            return False
        else:
            val = self.client.set_parameter(arguments[1], arguments[2])
            self.log.info('Return code {}'.format(val))
            return val

    def process_get(self, text):
        '''
            Process all get commands.
            Correct syntax are: get parameter
            Returns True if correct syntax and correct request
            Returns False if incorrect syntax and/or request
        '''
        arguments = text.split()
        if len(arguments) != 2:
            self.log.warn('get incorrect syntax')
            return False
        else:
            val = self.client.get_parameter(arguments[1])
            self.log.info('Return code {}'.format(val))
            return val

    def process_load(self, text):
        '''
            Process all load commands.
            Correct syntax are: load [weights]
            Returns True if correct syntax and correct request
            Returns False if incorrect syntax and/or request.
            Requests are incorrect when the loading of the specified weight file
            fails for the trained model
        '''
        arguments = text.split()
        if len(arguments) == 1:
            self.log.hist('Random Init of the weights')
            val = self.client.load_model_weights()
            return True
        else:
            val = self.client.load_model_weights(arguments[1])
            self.log.info('Return code {}'.format(val))
            return val





