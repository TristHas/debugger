##!/usr/bin/env python
## -*- coding: utf-8 -*-

from debugger.logging import Logger
from debugger.core.canvas import Canvas
from debugger.core.processor import Processor
from debugger.io import load_data
from debugger.conf import *
from debugger.models.models import LogisticModel, MLPModel
from debugger.models.trainer import NLL_Trainer
from debugger.network.client import RemoteClient, LocalClient
from vispy import app, use
use(app = 'PyQt5')
import Queue, sys

if __name__ == '__main__':
    app.set_interactive()

log = Logger(CTRL_LOG_FILE)
train_set, valid_set, test_set = load_data('mnist.pkl.gz')
transmit = Queue.Queue()
targets = {}
struct = {  0: [20, (28, 28)],
            1: [10, (4, 5)],
}

#model = LogisticModel(input_shape=(28, 28), n_out=10)
model = MLPModel(input_shape=(28, 28), n_hidden = 20, n_out=10)
trainer = NLL_Trainer(transmit, model, train_set, valid_set, test_set)
client = LocalClient(trainer)
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

def show(layer, cumul=None, nodeId = -1):
    if cumul is None:
        layers = [layer]
    else:
        layers = range(layer)
    target = (layer, cumul, nodeId)
    if not target in targets:
        add_target = all(client.add_target(x) for x in layers)
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

def quit():
    c.stop_running()
    exit()

def print_set():
    from matplotlib import pyplot as plt
    y = x.transpose().reshape(10, 28, 28)
    plt.imshow(y[2,:,:])
    plt.show()

x = train_set[0].get_value()[:20,:].T
dico = {0:x}
#transmit.put(dico)
print 'Ready!'






