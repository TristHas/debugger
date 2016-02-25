#!/usr/bin/env python
# -*- coding: utf-8 -*-

def main_without_lasagne():
    train_set, valid_set, test_set = load_data('mnist.pkl.gz')
    learning_rate       = 0.13
    labels              = T.ivector('y')
    index               = T.lscalar()
    batch_size          = 500
    model               = LogisticModel(input_shape=(28, 28), n_out=10)
    cost                = -T.mean(T.log(model.output)[T.arange(labels.shape[0]), labels])
    g_params            = [T.grad(cost=cost, wrt=param) for param in model.params]
    updates             = [(param, param - learning_rate * g_param) for param, g_param in zip(model.params, g_params)]

    def errors():
        return T.mean(T.neq(model.pred, labels))

    def functions(model):
        train_model = theano.function(
            inputs  = [index],
            outputs = [cost],
            updates = updates,
            givens  = {
                model.input: train_set[0][index * batch_size: (index + 1) * batch_size],
                labels: train_set[1][index * batch_size: (index + 1) * batch_size]
            }
        )

        test_model = None
        test_model = theano.function(
            inputs=[],
            outputs=errors(),#T.mean(T.neq(model.pred, labels)),
            givens={
                model.input: test_set[0],
                labels: test_set[1]
            }
        )
        validate_model = theano.function(
            inputs=[],
            outputs=errors(),#T.mean(T.neq(model.pred, labels)),
            givens={
                model.input: valid_set[0],
                labels: valid_set[1]
            }
        )
        return train_model, validate_model, test_model


    def build_mlp(input_var=None):
        l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                         input_var=input_var)

        l_hid1 = lasagne.layers.DenseLayer(
                l_in_drop, num_units=20,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform(),
                name = 0)
        l_out = lasagne.layers.DenseLayer(
                l_hid2_drop, num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax,
                name=1)
        return l_out
    log                = Logger(CTRL_LOG_FILE)
    transmit           = Queue.Queue()
    transmitter        = Transmitter(transmit)
    train, valid, test = functions(model)
    state              = TrainerState()
    trainer            = Trainer(train, valid, test, state, transmitter)
    global ctrl
    ctrl               = TrainerCtrl(state, trainer)



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
struct = {  1: [20, (28, 28)],
            2: [10, (4, 5)],
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

def show(layer, cumul=None, nodeId = -1, gradient = None):
    if cumul is None:
        layers = [str(layer) + '.W']
    else:
        # Should be in a special function, too sensitive to little changes in
        l_tmp = [str(l) + '.W', str(l) + '.b' for range(1, layer+1)]
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

x = train_set[0].get_value()[:20,:].T
dico = {0:x}
#transmit.put(dico)
print 'Ready!'






