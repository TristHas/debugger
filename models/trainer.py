#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..debug.util.conf import *
from ..debug.util.helpers import Logger
from ..debug.util.mnist_loader import load_data
from models import LogisticModel

import theano.tensor as T
import theano

class NLL_Trainer(object):
    def __init__(self, model, train_set, valid_set, test_set, cost = self.nll, test_func = self.errors):

        self.model = model
        self.test_model = theano.function(
            inputs=[index],
            outputs=test_func(y),
            givens={
                x: test_set[0][index * batch_size: (index + 1) * batch_size],
                y: test_set[1][index * batch_size: (index + 1) * batch_size]
            }
        )

        self.validate_model = theano.function(
            inputs=[index],
            outputs=test_func(y),
            givens={
                x: valid_set[0][index * batch_size: (index + 1) * batch_size],
                y: valid_set[1][index * batch_size: (index + 1) * batch_size]
            }
        )

        g_W = T.grad(cost=cost, wrt=model.W)
        g_b = T.grad(cost=cost, wrt=model.b)

        updates = [(classifier.W, classifier.W - learning_rate * g_W),
                   (classifier.b, classifier.b - learning_rate * g_b)]

        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set[0][index * batch_size: (index + 1) * batch_size],
                y: train_set[1][index * batch_size: (index + 1) * batch_size]
            }
        )


    def nll(self, y):
        assert
        return -T.mean(T.log(self.model.output)[T.arange(y.shape[0]), y])

    def errors(self, y):
        assert y.ndim == self.pred.ndim
        return T.mean(T.neq(self.pred, y))

if __name__ == '__main__':
    datasets = load_data('mnist.pkl.gz')
    batch_size=600
    train_set = datasets[0]
    valid_set = datasets[1]
    test_set = datasets[2]

    model = LogisticModel(input=x, input_shape=28 * 28, n_out=10)
    trainer = NLL_Trainer(model, train_set, valid_set, test_set)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels









