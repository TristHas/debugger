#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import numpy as np
from theano import tensor as T


rng = np.random.RandomState(1234)
###
###     All Layers should provide:
###     get_weights, get_bias, and standard_init methods
class baseLayer(object):
    def __init__(self):
        pass

    def get_weights(self):
        """
            DOC
        """
        return self.W.get_value()

    def get_biass(self):
        """
            DOC
        """
        return self.b.get_value()

    def standard_init(self):
        """
            DOC
        """
        try:
            W = np.asarray(
                rng.uniform(
                    low=  -np.sqrt(6. / (self.dim[0] + self.dim[1])),
                    high= np.sqrt(6. /  (self.dim[0] + self.dim[1])),
                    size=(self.dim[0], self.dim[1])
                 ),
                dtype=theano.config.floatX
            )
            b =np.zeros(
                (self.dim[1],),
                dtype=theano.config.floatX
            )
            self.b.set_value(b)
            self.W.set_value(W)
        except IndexError as e:
            return False
        return True

class fullyConnectedLayer(baseLayer):
    """
        DOC
    """
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        super(fullyConnectedLayer, self).__init__()
        self.layerType = 'full'
        self.dim = (n_in, n_out)
        self.input = input
        self.W = theano.shared(
            value=np.zeros(
                (self.dim[0], self.dim[1]),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros(
                (self.dim[1],),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.standard_init()
        ### Try and see what happens if we put those 2 lines in the __init__
        self.params = [self.W, self.b]
        self.output = T.dot(self.input, self.W) + self.b

class logRegLayer(baseLayer):
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """


    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        super(logRegLayer, self).__init__()
        self.layerType = 'logreg'
        self.dim = (n_in, n_out)
        self.input = input
        self.W = theano.shared(
            value=np.zeros(
                (self.dim[0], self.dim[1]),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros(
                (self.dim[1],),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        ### Try and see what happens if we put those 2 lines in the __init__
        self.params = [self.W, self.b]
        self.output = T.nnet.softmax(T.dot(self.input, self.W) + self.b)

