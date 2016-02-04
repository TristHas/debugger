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
#       - input
#       - output
#       - dim

class LogisticModel(object):
    def __init__(self, input_shape, n_out):
        '''
            DOC
        '''
        self.log = Logger(MODEL_LOG_FILE)
        n_in = np.prod(input_shape)
        self.input = T.matrix('x')

        # Build model architecture
        ####
        ####        ARCH
        ####
        self.l_0 = fullyConnectedLayer(self.input, n_in, n_out)
        self.struct = {0: [False, self.l_0, self.l_0.layerType, (n_in, n_out)]}
        # Parameters
        self.params = []
        for layer in self.struct:
            self.params.extend(self.struct[layer][1].params)

        # Selects the index of the highest output
        self.output = self.l_0.output
        self.pred = T.argmax(self.output, axis=1)

    def add_target(self, target = None):
        '''
            DOC
        '''
        if target in self.struct:
            self.struct[target][0] = True
            self.log.info('added target {}'.format(target))
            return True
        else:
            self.log.warn('asked to add target {} which does not exist'.format(target))
            return False

    def remove_target(self, target):
        '''
            DOC
        '''
        if target in self.struct:
            self.struct[target][0] = False
            self.log.info('added target {}'.format(target))
            return True
        else:
            self.log.warn('asked to add target {} which does not exist'.format(target))
            return False

    def drop_weights(self):
        '''
            DOC
        '''
        dico = {}
        for layer in self.struct:
            if self.struct[layer][0] == True:
                self.get_layer_weight(layer, dico)
        return dico

    def get_layer_weight(self, layer, dico):
        '''
            DOC
        '''
        weights = self.struct[layer][1].get_weights()
        dico[layer]=weights

    def get_struct(self):
        '''
            DOC
        '''
        return self.struct

    def load_weights(self, file = None):
        '''
            DOC
        '''
        if file is None:
            return self.standard_weight_init()
        else:
            #TODO Weight loading. How shall we serialize? cPickle a dict?
            return False

    def standard_weight_init(self):
        '''
            DOC
        '''
        return all( self.struct[layer][1].standard_init() for layer in self.struct)
