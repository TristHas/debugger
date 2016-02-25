#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import timeit, time
import threading, Queue

import lasagne
import theano.tensor as T
import theano
import numpy

from debugger.conf import *
from debugger.util.math import *
from debugger.logging import Logger
from debugger.perf import Timer
from models import LogisticModel

log = Logger(TRAINER_LOG_FILE, level=V_DEBUG)

class TrainerCtrl(object):
    def __init__(self, model_builder, sm):
        """
            DOC
        """
        input_var = T.matrix('inputs', dtype='float32')
        target_var = T.ivector('targets')
        model = model_builder(input_var)

        train_prediction    = lasagne.layers.get_output(model)
        train_losses        = lasagne.objectives.categorical_crossentropy(train_prediction, target_var)
        train_loss          = train_losses.mean()
        params              = lasagne.layers.get_all_params(model, trainable=True)
        updates             = lasagne.updates.nesterov_momentum(train_loss, params, learning_rate=0.01, momentum=0.9)
        # Training
        train_fn            = theano.function([input_var, target_var], train_loss, updates=updates)
        test_prediction     = lasagne.layers.get_output(model, deterministic=True)
        test_loss           = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_acc            = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

        # Testing and Validation
        test_loss           = test_loss.mean()
        valid_fn            = theano.function([input_var, target_var], test_acc)
        test_fn             = valid_fn

        transmit            = Queue.Queue()
        self.transmitter    = Transmitter(transmit, model)
        self.state          = TrainerState()
        self.trainer        = Trainer(train_fn, valid_fn, test_fn, self.state, self.transmitter, sm)
        self.model          = model

        self.weight_orders  = {}
        self.grad_orders    = {}


    def new_order(self, target, layers):
        '''
            DOC
        '''
        # Be carefull about access concurrency, shouldn't need a lock here?
        log.info('New order: target = {}, layers = {}'.format(target, layers))
        target_type = self.check_target(target)
        if target_type == 'weight':
            if target not in self.weight_orders:
                self.weight_orders[target]  = layers
                log.verb('Weight orders are: {}'.format(self.weight_orders))
                self.state.weight_state     = union(self.weight_orders)
                self.update_weight_drop()
            else:
                return False
        elif target_type == 'grad':
            if target not in self.grad_orders:
                self.grad_orders[target] = layers
                self.state.grad_state    = union(self.grad_orders)
                self.update_grad_drop()
            else:
                return False
        else:
            return False
        return True

    def cancel_order(self, target):
        '''
            DOC
        '''
        # Be carefull about state access concurency
        target_type = self.check_target(target)
        if target_type == 'weight':
            if target in self.weight_orders:
                del self.weight_orders[target]
                self.state.weight_state     = union(self.weight_orders)
                self.update_weight_drop()
            else:
                return False
        elif target_type == 'grad':
            if target not in self.grad_orders:
                del self.grad_orders[target]
                self.state.grad_state       = union(self.grad_orders)
                self.update_grad_drop()
            else:
                return False
        else:
            return False
        return True

    def load_weight(self, weight = None):
        '''
            CODE
        '''
        pass

    def start(self):
        self.trainer.start()

    def stop(self):
        self.trainer.stop()

    def update_weight_drop(self):
        '''
            Now drops both wieght and bias
        '''
        log.debug('Weight states are: {}'.format(self.state.weight_state))
        layers = lasagne.layers.get_all_layers(self.model)
        for i in range(1, len(layers)):
            for key in layers[i].params:
                log.debug('Considering key param {}'.format(key))
                if key.name in self.state.weight_state:    # Tag parameters as record
                    layers[i].params[key].add('recording')
                    log.debug('Tagged layer {} to record'.format(layers[i].params[key]))
                else:                               # Those parameters are not tagged as recording
                    try:
                        layers[i].params[key].remove('recording')
                        log.debug('Untagged layer {} from record'.format(layers[i].params[key]))
                    except KeyError:
                        pass

    def update_grad_drop(self):
        pass

    def check_target(self, target):
        '''
            Returns weight if target is weight type and correct with model
            Returns grad if target is grad type and correct with model
            Returns False otherwise
        '''
        if target[3]:
            return 'grad'
        else:
            return 'weight'

    def update_trainer(self):
        # Set trainer's train function based on gradient situation
        # Set trainer's settings       based on his own
        pass



class Transmitter(object):
    def __init__(self, transmit, model):
        self.transmit   = transmit
        self.model      = model

    ### Available to trainer
    def train_results(self, results):
        # Get weight data
        result_dic = {}
        self.drop_weight(result_dic)
        self.drop_grad(result_dic, results)
        self.transmit.put(result_dic)
        log.debug('Dropped {}'.format(result_dic))
        pass

    def valid_results(self, results):
        log.info('validation accuracy %f %%' % (results * 100))

    def test_results(self, results):
        log.info('test accuracy %f %%' % (results * 100))

    ### Internal helpers
    def drop_weight(self, dico):
        layers = lasagne.layers.get_all_layers(self.model)
        for i in range(1, len(layers)):
            params = layers[i].get_params(recording=True)
            for param in params:
                dico[param.name] = param.get_value()

    def drop_grad(self, dico, data):
        pass

class TrainerState(object):
    def __init__(self):
        self.log            = Logger(TRAINER_LOG_FILE, level=V_DEBUG)
        self.run_state      = 'idle'
        self.state_lock     = threading.Lock()
        self.grad_state     = []
        self.weight_state   = []
        self.training_lock  = threading.Lock()

class Trainer(object):
    def __init__(self, train, validate, test, state, transmitter, sm):
        self.state       = state
        self.train       = train
        self.validate    = validate
        self.test        = test
        self.sm          = sm
        # Should be set by controler.
        self.transmitter = transmitter
        self.settings = {
            'batch_size'           : 500, # Particulier. Peut dépendre des fonctions
                                          # quand on travaille avec des convolutions
            'n_batch'              : 100,
            'f_validation'         : 100,
            'f_record'             : 100,
            'n_epoch'              : 100,
        }
        self.timing   = None

    def start(self):
        with self.state.state_lock:
            if self.state.run_state == 'idle':
                self.thread = threading.Thread(target = self._training_process, name = 'training process', args = ())
                self.thread.start()
                self.state.run_state = 'running'

    def stop(self):
        with self.state.state_lock:
            if self.state.run_state == 'running':
                self.state.run_state = 'asked_stop'

    def _training_process(self):
        epoch = 0
        while (epoch < self.settings['n_epoch']):
            log.info('Epoch %i' % (epoch + 1))
            i = 0
            for batch in self.sm.iterate_minibatch('train', self.settings['batch_size']):
                with self.state.training_lock:
                    inputs, indices = batch
                    batch_index = epoch * self.settings['n_batch'] + i
                    train_results = self.train(inputs, indices)

                    if batch_index % self.settings['f_validation'] == 0:
                        inputs, indices = self.sm.get_whole_set('valid')
                        validation_results = self.validate(inputs, indices)
                        self.transmitter.valid_results(validation_results)

                    if batch_index % self.settings['f_record'] == 0:
                        self.transmitter.train_results(train_results)
                    i += 1
            epoch += 1
            with self.state.state_lock:
                if self.state.run_state == 'asked_stop':
                    break
            #? automatic stop?
            #? automatic training parameter update?
        inputs, indices = self.sm.get_whole_set('test')
        test_results = self.test(inputs, indices)
        self.transmitter.test_results(test_results)
        with self.state.state_lock:
            self.state.run_state = 'idle'

