#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Queue
import time

import theano.tensor as T
import numpy

from debugger.debug.util.mnist_loader import load_data
from debugger.models.models import LogisticModel
from debugger.models.trainer import NLL_Trainer

if __name__ == '__main__':
    datasets = load_data('mnist.pkl.gz')
    train_set = datasets[0]
    valid_set = datasets[1]
    test_set = datasets[2]

    transmit = Queue.Queue()

    model = LogisticModel(input_shape=28 * 28, n_out=10)
    trainer = NLL_Trainer(transmit, model, train_set, valid_set, test_set)
    trainer.launch_training()
    time.sleep(10)

