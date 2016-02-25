#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lasagne
import theano

from debugger.logging import Logger
from debugger.conf import *
from debugger.io import load_mnist, SetManager
from debugger.models.trainer import TrainerCtrl
from debugger.network.client import LocalClient
from debugger.core.canvas import Canvas
from debugger.core.processor import Processor


import sys

def main_with_lasagne():
    def build_mlp(input_var=None):
        l_in = lasagne.layers.InputLayer(shape=(None, 784),
                                         input_var=input_var)
        l_hid1 = lasagne.layers.DenseLayer(
                l_in, num_units=20,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform(),
                name = 1)
        l_out = lasagne.layers.DenseLayer(
                l_hid1, num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax,
                name = 2)
        return l_out

    global sm
    sm = SetManager(load_mnist())
    global ctrl
    ctrl = TrainerCtrl(build_mlp, sm)

global ctrl
main_with_lasagne()

transmit = ctrl.transmitter.transmit
targets = {}
struct = {  1: [20, (28, 28)],
            2: [10, (4, 5)],
}

log = Logger(CTRL_LOG_FILE)
client = LocalClient(ctrl)
p = Processor(targets, struct)
c = Canvas(transmit, p)

###
###     Aliases
###
lr = 'learning_rate'
bs = 'batch_size'

###
###     Commands
###
def get(param):
    return client.get_parameter(param)

def set(param, value):
    client.set_parameter(param, value)

def start():
    client.start_training()

def stop():
    client.stop_training()

def pause():
    client.pause_training()

def resume():
    client.resume_training()

def load(weight_file = None):
    return client.load_model_weights(weight_file)

def show(layer, cumul=None, nodeId = -1, gradient = None):
    if cumul is None:
        layers = [str(layer) + '.W']
    else:
        # Should be in a special function, too sensitive to little changes in
        l_tmp = [[str(l) + '.W', str(l) + '.b'] for l in range(1, layer+1)]
        layers = list(chain.from_iterable(l_tmp))
        print layers
    target = (layer, cumul, nodeId, gradient)
    if not target in targets:
        add_target = client.add_target(target, layers)
        if add_target:
            order_success = p.order(target, layers)
            if order_success:
                client.start_record()
                log.info('Successfully added target {}'.format(target))
            else:
                ##TODO: Should remove targets from client
                log.error("Processor Order failed")
        else:
            ##TODO: Should remove targets from client
            log.error('Client Target Add Failed for  {}'.format(target))
    else:
        log.error('Target {} asked whereas allready in target control list'.format(target))

def hide(layer, cumul=None, nodeId = -1, gradient = None):
    ###TODO
    pass

def quit():
    c.stop_running()
    exit()

def print_set():
    from matplotlib import pyplot as plt
    y = x.transpose().reshape(10, 28, 28)
    plt.imshow(y[2,:,:])
    plt.show()

#x = train_set[0].get_value()[:20,:].T
#dico = {0:x}
print 'Ready!'
