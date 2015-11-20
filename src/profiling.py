#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cProfile
#from printer import main
from theano_mnist_lgd import sgd_optimization_mnist

if __name__ == '__main__':
    cProfile.run('sgd_optimization_mnist()')
