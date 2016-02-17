#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import timeit, time
import threading
import theano.tensor as T
import theano
import numpy
from ..debug.util.conf import *
from ..debug.util.helpers import Logger, WithTimer, Timer
from models import LogisticModel


class NLL_Trainer(object):
    """
        DOC
    """
    def __init__(   self, transmit, model, train_set, valid_set, test_set,
                    batch_size = 600, learning_rate = 0.13, cost = None, test_func = None,
                    regularization_factor = 0
                ):
        """
            DOC
        """
        self.log            = Logger(TRAINER_LOG_FILE, level=V_DEBUG)
        self.transmit       = transmit
        self.is_recording   = False
        self.is_paused      = False
        self.is_running     = False

        self.labels = T.ivector('y')        # labels, presented as 1D vector of [int] labels
        self.index = T.lscalar()            # index to a minibatch
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

        self.batch_size = theano.shared(0)
        self.reg        = theano.shared(regularization_factor)
        self.set_batch_size(batch_size)

        self.validation_frequency = self.n_train_batches
        self.record_frequency     = self.n_train_batches
        self.improvement_step_threshold = 0.0001
        self.n_epochs = 1000

        self.learning_rate = theano.shared(learning_rate)

        if not test_func:
            test_func = self._errors

        self.model = model
        self.nll  = -T.mean(T.log(self.model.output)[T.arange(self.labels.shape[0]), self.labels])
        if not cost:
            cost = self.nll
        else:
            cost = cost + self.reg #* racine(somme(Poids au carrés))

        g_params  = [T.grad(cost=cost, wrt=param) for param in self.model.params]
        updates   = [(param, param - self.learning_rate * g_param) for param, g_param in zip(self.model.params, g_params)]

        self.train_model = theano.function(
            inputs=[self.index],
            outputs=[cost] + g_params,
            #outputs = cost,
            updates=updates,
            givens={
                model.input: self.train_set[0][self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.labels: self.train_set[1][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        self.test_model = theano.function(
            inputs=[self.index],
            outputs=test_func(),
            givens={
                model.input: self.test_set[0][self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.labels: self.test_set[1][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        self.validate_model_2 = theano.function(
            inputs=[self.index],
            outputs=test_func(),
            givens={
                model.input: self.valid_set[0][self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.labels: self.valid_set[1][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        self.validate_model_1 = theano.function(
            inputs=[],
            outputs=test_func(),
            givens={
                model.input: self.valid_set[0],
                self.labels: self.valid_set[1]
            }
        )


    ###
    ###     Non client-accessible functions
    ###
    def _errors(self):
        '''
            Ratio of errors in the prediction
        '''
        assert self.labels.ndim == self.model.pred.ndim
        return T.mean(T.neq(self.model.pred, self.labels))

    def _training_process(self):
        """
            Training loop
        """
        timer = Timer()
        minibatch_avg_cost = 0
        best_validation_loss = numpy.inf
        test_score = 0.
        # time ne prends pas en compte les pauses/resume
        start_time = timeit.default_timer()
        epoch = 0

        while (epoch < self.n_epochs) and self.is_running:
            while self.is_paused:
                time.sleep(1)
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):                            # minibatch_index within an epoch
                res = timer.time(self.train_model, minibatch_index)
                cost = res[0]
                #print res[1].shape
                #self.transmit.put({0:res[1] * 10})
                minibatch_avg_cost += cost
                iter = (epoch - 1) * self.n_train_batches + minibatch_index                 # Iter = number of minibatch passed
                if (iter + 1) % self.record_frequency == 0 and self.is_recording:
                    data = self.model.drop_weights()
                    #print type(data)
                    #print len(data)
                    #print data[0].shape
                    if data:
                        self.transmit.put(data)
                    else:
                        print 'WHAT WENT WRONG? EMPTY DATA DROPPEND BY MODEL IN TRAINER'

                if (iter + 1) % self.validation_frequency == 0:
                    training_score = minibatch_avg_cost / self.validation_frequency
                    minibatch_avg_cost = 0
                    this_validation_loss = timer.time(self.validate_model_1)
                    self.log.info(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            self.n_train_batches,
                            this_validation_loss * 100.
                        )
                    )
                    self.log.verb(
                        'epoch %i, minibatch %i/%i, training scor %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            self.n_train_batches,
                            training_score
                        )
                    )


                    if this_validation_loss < best_validation_loss or True:
                        best_validation_loss = this_validation_loss
                    else:
                        self.is_running = False
                        test_losses = [self.test_model(i) for i in xrange(self.n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        self.log.info(
                            (
                                '     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                self.n_train_batches,
                                test_score * 100.
                            )
                        )

        end_time = timeit.default_timer()
        self.log.info(
            (
                'Optimization complete with best validation score of %f %%,'
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., test_score * 100.)
        )
        self.log.info('The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time)))
        self.log.info('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.1fs' % ((end_time - start_time)))

        print 'Validation: avg time = {}  ||  total time = {}'.format(timer.get_avg_time(self.validate_model_1), timer.get_total_time(self.validate_model_1))
        print 'Training: avg time = {}  ||  total time = {}'.format(timer.get_avg_time(self.train_model), timer.get_total_time(self.train_model))

        ###
        ###     Should any value be reinitalised?
        ###
    def set_batch_size(self, batch_size):
        self.batch_size.set_value(batch_size)
        self.n_train_batches = self.train_set[0].get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = self.valid_set[0].get_value(borrow=True).shape[0] / batch_size
        self.n_test_batches = self.test_set[0].get_value(borrow=True).shape[0] / batch_size



    ###
    ###     Client-accessible functions
    ###
    def start_training(self):
        '''
        Runs the training process in a woker thread.
        Two steps:
            - Initialisation des valeurs de training (best validation loss et test_score).
            - Training loop.
        '''
        self.is_running = True
        thread = threading.Thread(target = self._training_process, name = 'training process', args = ())
        thread.start()

    def stop_training(self):
        '''
            Terminate the training process in a clean manner.
            Si le training est pausé, ne fait rien.
            Affiche les scores et temps en fin de Thread
        '''
        if (not self.is_paused and self.is_running):
            self.is_running = False
            return True
        else:
            return False

    def pause_training(self):
        '''
            If the training is running and not paused,
            this method returns true and induces an Idle
            Loop with 1 s frequency until resume_training
            is called. Otherwise, returns False
        '''
        if not self.is_running or self.is_paused:
            return False
        self.is_paused  = True
        return True

    def resume_training(self):
        '''
            If the training is running and paused, this method
            ends Idle Loop and returns True.
            Other wise does nothing and returns False
        '''
        if self.is_paused and self.is_running:
            self.is_paused  = False
            return True
        else:
            return False

    def add_target(self, target = None):
        '''
            Set the target layer as to be recorded. See model.target method
        '''
        return self.model.add_target(target)

    def remove_target(self, target = None):
        '''
            Set the target layer as not to be recorded. See model.target method
        '''
        return self.model.remove_target(target)

    def start_record(self):
        '''
            Asks model to stop putting weights in transmit Queue
        '''
        self.is_recording = True

    def stop_record(self):
        '''
            Asks model to stop putting weights in transmit Queue
        '''
        self.is_recording = False

    def load_model_weights(self, weight_file = None):
        '''
            Sets models weights to the values serialized in weight_file
        '''
        if not self.is_running or self.is_paused:
            ret_val = self.model.load_weights(weight_file)
        else:
            self.log.error('Asked for weight loading while training is running')
            return False

    def set_parameter(self, parameter, value):
        '''
            Set trainer's parameter to value and returns value.
            If parameter is not an attribute of trainer, returns None.
        '''
        self.log.debug('Setting Parameter {} to {}'.format(parameter, value))
        try:
            if parameter == 'batch_size':
                val = int(value)
                self.set_batch_size(val)
            elif parameter == 'learning_rate':
                val = float(value)
                self.learning_rate.set_value(val)
            else:
                val = float(value)
                setattr(self, parameter, val)
            new_val = getattr(self, parameter)
            self.log.hist('Has set {} to {}'.format(parameter, new_val))
            return new_val
        except (ValueError, AttributeError) as e:
            self.log.error('Failed to set {} as {}. Error is {}'.format(parameter, val, e))
            return None

    def get_parameter(self, parameter):
        '''
            Returns the value of trainer's parameter.
            If parameter is not an attribute of trainer,
            returns None
        '''
        try:
            val = getattr(self, parameter)
            self.log.info('Got parameter {}={}'.format(parameter, val))
            return val
        except AttributeError as e:
            self.log.error('Error Getting parameter {}: {}'.format(parameter, e))
            return None


#    def drop_input(self, label):
#        data = train_set[0].get_value()[10,:]

