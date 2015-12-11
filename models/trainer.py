#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..debug.util.conf import *
from ..debug.util.helpers import Logger
from models import LogisticModel

import theano.tensor as T
import theano

class NLL_Trainer(object):
    def __init__(self, model, train_set, valid_set, test_set, index, x, y, batch_size, learning_rate, cost = None, test_func = None):
        """
            Trainer parameters:
                training parameters:
                    self.batch_size
                    self.learnin_rate

                self.train_set
                self.valid_set
                self.test_set

                self.index
                self.model
                self.x
                self.y
                self.test_func

        """

        if not test_func:
            test_func = self.errors
        self.model = model
        self.nll  = -T.mean(T.log(self.model.output)[T.arange(y.shape[0]), y])
        if not cost:
            cost = self.nll

        g_W = T.grad(cost=cost, wrt=model.W)
        g_b = T.grad(cost=cost, wrt=model.b)

        updates = [(model.W, model.W - learning_rate * g_W),
                   (model.b, model.b - learning_rate * g_b)]

        self.train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set[0][index * batch_size: (index + 1) * batch_size],
                y: train_set[1][index * batch_size: (index + 1) * batch_size]
            }
        )

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

    def errors(self, y):
        assert y.ndim == self.model.pred.ndim
        return T.mean(T.neq(self.model.pred, y))










