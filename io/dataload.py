#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle, gzip
import theano
import theano.tensor as T
import numpy as np
from debugger.conf import *

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    #############
    # LOAD DATA #
    #############

    data_file = os.path.join(DATA_DIR, dataset)
    if not os.path.isfile(data_file):
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, data_file)

    print '... loading data'
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_mnist(dataset= 'mnist.pkl.gz'):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    data_file = os.path.join(DATA_DIR, dataset)
    if not os.path.isfile(data_file):
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, data_file)

    # Should Not work with it saved as gz
    print '... loading data'
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    sets = {
        'train': train_set,
        'valid': valid_set,
        'test' : test_set
    }
    return sets

def init_dataset(data_xy, borrow = True):
    # What's borrow again?
    data_x, data_y = data_xy
    set_size = data_x.shape[0]
    assert data_y.shape[0] == set_size
    #shared_x = theano.shared(np.asarray(data_x,
    #                                    dtype=theano.config.floatX),
    #                                    borrow=borrow)
    #shared_y = theano.shared(np.asarray(data_y,
    #                                    dtype=theano.config.floatX),
    #                                    borrow=borrow)
    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.int32)
    score = np.zeros((set_size,))
    repet = np.inf * np.ones((set_size,))
    return data_x, data_y, score, repet

class SetManager(object):
    '''
        DOC
    '''
    def __init__(self, sets):
        '''
            The setManager takes a list of set as input.
            These sets should be [dataAsnpArray, nplabelAsnpArray]
        '''
        self.sets = {}
        for set_name in sets:
            self.sets[set_name] = init_dataset(sets[set_name])

    def iterate_minibatch(self, set, batchsize, i=0, shuffle=False):
        set_size = self.sets[set][1].shape[0]
        if shuffle:
            indices = np.arange(set_size)
            np.random.shuffle(indices)
        for start_idx in range(0, set_size - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield self.sets[set][0][excerpt], self.sets[set][1][excerpt]

    def get_whole_set(self, set):
        return self.sets[set][0], self.sets[set][1]
