#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import timeit
import sys

from debugger.debug.util.mnist_loader import load_data
from debugger.models.models import LogisticModel
from debugger.models.trainer import NLL_Trainer
import theano.tensor as T
import numpy

if __name__ == '__main__':
    datasets = load_data('mnist.pkl.gz')
    train_set = datasets[0]
    valid_set = datasets[1]
    test_set = datasets[2]

    batch_size=600
    learning_rate=0.13



    n_train_batches = train_set[0].get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set[0].get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set[0].get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels




    model = LogisticModel(input=x, input_shape=28 * 28, n_out=10)
    trainer = NLL_Trainer(model, train_set, valid_set, test_set, index, x, y, batch_size, learning_rate, cost = None, test_func = None)

    validation_frequency = n_train_batches
    improvement_step_threshold = 0.0001
    n_epochs=1000
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = trainer.train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [trainer.validate_model(i) for i in xrange(n_valid_batches)]
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
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss > best_validation_loss - improvement_step_threshold:
                        done_looping = True

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [trainer.test_model(i) for i in xrange(n_test_batches)]
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

                    # save the best model

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
