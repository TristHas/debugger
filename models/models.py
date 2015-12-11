#!/usr/bin/env python
# -*- coding: utf-8 -*-
from layers import fullyConnectedLayer
import numpy as np
from ..debug.util.helpers import Logger
from ..debug.util.conf import *
from theano import tensor as T


#   A model should define the following functions:
#
#       - add_targets
#       - remove_targets
#       - drop_weights
#
#   Furthermore, it should present the following arguments
#       - W
#       - b


class LogisticModel(object):
    def __init__(self, input_shape, n_out):
        self.log = Logger(SGD_LOG_FILE)#, V_DEBUG)
        n_in = np.prod(input_shape)
        self.input = T.matrix('x')

        # Build model architecture
        ####
        ####        ARCH
        ####
        self.l_1 = fullyConnectedLayer(self.input, n_in, n_out)
        # Only one layer here

        ####
        ####        WEIGHT
        ####
        self.W = self.l_1.W
        self.b = self.l_1.b

        weight_shape = input_shape
        self.struct = {'l_1': [False, self.l_1, weight_shape]}

        # Selects the index of the highest p_y_given_x
        self.output = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
        self.pred = T.argmax(self.output, axis=1)

    def add_targets(self, targets = None):
        for tagret in targets:
            if target in self.struct:
                self.struct[target][0] = True
                self.log.info('added target {}'.format(target))
            else:
                self.log.warn('asked to add target {} which does not exist'.format(target))

    def remove_targets(self, targets):
        for tagret in targets:
            if target in self.struct:
                self.struct[target][0] = False
                self.log.info('added target {}'.format(target))
            else:
                self.log.warn('asked to add target {} which does not exist'.format(target))

    def drop_weights(self):
        dico = {}
        for layer in self.struct:
            if self.struct[layer][0] == True:
                ### Should reshape weights here?
                dico[layer] = self.struct[layer][1].get_weights().reshape(self.struct[layer][2])
        return dico

