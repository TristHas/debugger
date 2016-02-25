#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This is used in theano tutorials.
# Why do they loop over minibatches? We should time it with our
# funtion to see if it's oddly faster or try to understand why
self.validate_model_2 = theano.function(
    inputs=[self.index],
    outputs=test_func(),
    givens={
        model.input: self.valid_set[0][self.index * self.batch_size: (self.index + 1) * self.batch_size],
        self.labels: self.valid_set[1][self.index * self.batch_size: (self.index + 1) * self.batch_size]
    }
)
