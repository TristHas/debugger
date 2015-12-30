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

    def add_targets(self, target = None):
        if target in self.struct:
            self.struct[target][0] = True
            self.log.info('added target {}'.format(target))
            return True
        else:
            self.log.warn('asked to add target {} which does not exist'.format(target))
            return False

    def remove_targets(self, target):
        if target in self.struct:
            self.struct[target][0] = False
            self.log.info('added target {}'.format(target))
            return True
        else:
            self.log.warn('asked to add target {} which does not exist'.format(target))
            return False

    def drop_weights(self):
        dico = {}
        for layer in self.struct:
            if self.struct[layer][0] == True:
                self.get_layer_weight(layer, dico)
        return dico

    def get_layer_weight(self, layer, dico):
        weights = self.struct[layer][1].get_weights()
        tmp = {}
        for i in range(weights.shape[1]):
            i_th_filter = weights[:,i]
            self.log.debug('[MAIN THREAD] i_th_filter type: {}'.format(type(i_th_filter)))
            self.log.debug('[MAIN THREAD] i_th_filter shape: {}'.format(i_th_filter.shape))
            tmp[str(i)] = i_th_filter.reshape(self.struct[layer][2])
            self.log.debug('[MAIN THREAD] print_dico[str(i)].shape: {}'.format(tmp[str(i)].shape))
        dico[layer]=tmp
