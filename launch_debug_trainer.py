##!/usr/bin/env python
## -*- coding: utf-8 -*-

import Queue, sys
import debugger
from debugger.debug.core.canvase import Canvas, Processor
from debugger.debug.core.control_window import ControlWindow
from debugger.debug.util.mnist_loader import load_data
from debugger.models.models import LogisticModel
from debugger.models.trainer import NLL_Trainer
from debugger.network.client import LightClient, LocalClient
from vispy import app, use
use(app = 'PyQt5')

###
###     Easy_client
###

def main():
    x = app.Application()
    qt_app = x.native
    sys.exit(qt_app.exec_())

if __name__ == '__main__':
    app.set_interactive()

transmit = Queue.Queue()
targets = {}
p = Processor(targets)
c = Canvas(transmit, p)
train_set, valid_set, test_set = load_data('mnist.pkl.gz')
model = LogisticModel(input_shape=(28, 28), n_out=10)
trainer = NLL_Trainer(transmit, model, train_set, valid_set, test_set)
client = LocalClient(trainer)
p.set_model_struct(model.struct)

###
###     Aliases
###
lr = 'learning_rate'
bs = 'batch_size'

###
###     Commands
###
def get(param):
    client.get_parameter(param)

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
    client.load_model_weights(weight_file)

def show(layer, cumul=None, nodeId = -1):
    if cumul is None:
        layers = [layer]
    else:
        layers = range(layer)
    target = (layer, cumul, nodeId)
    if not target in targets:
        add_target = all(client.add_target(x) for x in layers)
        if add_target:
            order_success = p.order(target)
            if order_success:
                targets[target] = layers
                client.start_record()
                print ('Successfully added target {}'.format(target))
            else:
                ##TODO: Should remove targets from client
                print ("Order failed")
        else:
            ##TODO: Should remove targets from client
            print ('Client Target Add Failed for  {}'.format(target))
    else:
        print ('Target {} asked whereas allready in target control list'.format(target))

