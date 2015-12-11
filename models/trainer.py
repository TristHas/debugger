#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import timeit
import threading

import theano.tensor as T
import theano
import numpy

from ..debug.util.conf import *
from ..debug.util.helpers import Logger
from models import LogisticModel


class NLL_Trainer(object):
    def __init__(self, transmit, model, train_set, valid_set, test_set, batch_size = 600, learning_rate = 0.13, cost = None, test_func = None):
        """
        """

        self.transmit   = transmit
        self.is_paused  = False

        self.labels = T.ivector('y')        # labels, presented as 1D vector of [int] labels
        self.index = T.lscalar()            # index to a [mini]batch
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.batch_size = batch_size

        if not test_func:
            test_func = self.errors
        self.model = model
        self.nll  = -T.mean(T.log(self.model.output)[T.arange(self.labels.shape[0]), self.labels])
        if not cost:
            cost = self.nll

        g_W = T.grad(cost=cost, wrt=model.W)
        g_b = T.grad(cost=cost, wrt=model.b)

        updates = [(model.W, model.W - learning_rate * g_W),
                   (model.b, model.b - learning_rate * g_b)]

        self.train_model = theano.function(
            inputs=[self.index],
            outputs=cost,
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

        self.validate_model = theano.function(
            inputs=[self.index],
            outputs=test_func(),
            givens={
                model.input: self.valid_set[0][self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.labels: self.valid_set[1][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )


    def training_process(self):
        ### Should be updated with changes in batch_size
        n_train_batches = self.train_set[0].get_value(borrow=True).shape[0] / self.batch_size
        n_valid_batches = self.valid_set[0].get_value(borrow=True).shape[0] / self.batch_size
        n_test_batches = self.test_set[0].get_value(borrow=True).shape[0] / self.batch_size

        self.validation_frequency = n_train_batches
        self.record_frequency     = n_train_batches
        self.improvement_step_threshold = 0.0001
        self.n_epochs=1000
        self.done_looping = False

        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = timeit.default_timer()
        epoch = 0

        while (epoch < self.n_epochs) and (not self.done_looping):
            while self.is_paused:
                time.sleep(1)
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = self.train_model(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % self.record_frequency == 0 and self.is_recording:
                    data = self.model.drop_weights()
                    if data:
                        self.transmit.put(data)

                if (iter + 1) % self.validation_frequency == 0:
                    validation_losses = [self.validate_model(i) for i in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )
                    ####
                    ####    ERASED DEBUGGER INFO
                    ####
                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss > best_validation_loss - self.improvement_step_threshold:
                            self.done_looping = True

                        best_validation_loss = this_validation_loss
                        test_losses = [self.test_model(i) for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print(
                            (
                                '     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_score * 100.
                            )
                        )

        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %f %%,'
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., test_score * 100.)
        )
        print 'The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.1fs' % ((end_time - start_time)))

    def launch_training(self):
        thread = threading.Thread(target = self.training_process, name = 'training process', args = ())
        thread.start()

    def pause_training(self):
        self.is_paused  = True

    def resume_training(self):
        self.is_paused  = False

    def errors(self):
        assert self.labels.ndim == self.model.pred.ndim
        return T.mean(T.neq(self.model.pred, self.labels))

    def add_targets(self, targets = None):
        self.model.add_targets(targets)

    def remove_targets(self, targets = None):
        self.model.remove_targets(targets)

    def start_record(self):
        self.is_recording = True

    def stop_record(self):
        self.is_recording = False


