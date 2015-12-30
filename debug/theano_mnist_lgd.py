"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

from util.conf import *
from util.helpers import Logger
from util.mnist_loader import load_data

import os
import sys
import timeit
import cPickle

import numpy
import theano
import theano.tensor as T


if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
log = Logger(SGD_LOG_FILE, V_WARN, real_time = False)

class LogisticRegression(object):
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
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.input = input
        self.params = [self.W, self.b]
        self.output = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
        self.pred = T.argmax(self.output, axis=1)


    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        if y.ndim != self.pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.pred, y))
        else:
            raise NotImplementedError()


def sgd_optimization_mnist(transmit, learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    cost = classifier.negative_log_likelihood(y)
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found

    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant

    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0


    ###
    ###     Debug var init
    ###

    tmp_w_print = classifier.W.get_value()
    tmp_b_print = classifier.b.get_value()

    im_index = 0
    im_shape = (28,28)
    w_shape = im_shape

    log.debug('[MAIN THREAD] tmp_w_print type: {}'.format(type(tmp_w_print)))
    log.debug('[MAIN THREAD] tmp_w_print shape: {}'.format(tmp_w_print.shape))
    log.debug('[MAIN THREAD] tmp_b_print type: {}'.format(type(tmp_b_print)))
    log.debug('[MAIN THREAD] tmp_b_print shape: {}'.format(tmp_b_print.shape))

    print_dico = {}
    for i in range(tmp_w_print.shape[1]):
        i_th_filter = tmp_w_print[:,i]
        log.debug('[MAIN THREAD] i_th_filter type: {}'.format(type(i_th_filter)))
        log.debug('[MAIN THREAD] i_th_filter shape: {}'.format(i_th_filter.shape))
        print_dico[str(i)] = i_th_filter.reshape(w_shape)
        log.debug('[MAIN THREAD] print_dico[str(i)].shape: {}'.format(print_dico[str(i)].shape))

    init_dico = {}
    for keys in print_dico:
        init_dico[keys] = print_dico[keys].shape
    #print_data = fast_init_image(init_dico)
    #print_image_fast(print_dico, print_data)


    #reshaped_im = train_array[im_index].reshape(im_shape)

    #log.debug('[MAIN THREAD] reshaped_im type: {}'.format(type(reshaped_im)))
    #log.debug('[MAIN THREAD] train_array[0] shape: {}'.format(reshaped_im.shape))

    #print_dico = {'test_image': reshaped_im}
    #init_dico = {'test_image': reshaped_im.shape}
    #print_data = fast_init_image(init_dico)
    #print_image_fast(print_dico, print_data)



    while (epoch < n_epochs) and (not done_looping):
        
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )
                tmp_w_print = classifier.W.get_value()
                for i in range(tmp_w_print.shape[1]):
                    i_th_filter = tmp_w_print[:,i]
                    log.debug('[MAIN THREAD] i_th_filter type: {}'.format(type(i_th_filter)))
                    log.debug('[MAIN THREAD] i_th_filter shape: {}'.format(i_th_filter.shape))
                    print_dico[str(i)] = i_th_filter.reshape(w_shape)
                    log.debug('[MAIN THREAD] print_dico[str(i)].shape: {}'.format(print_dico[str(i)].shape))

                transmit.put(print_dico)
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'w') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = cPickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.pred)

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_values


def test():
    data_file='mnist.pkl.gz'
    im_index = 0
    im_shape = (28,28)

    dataset = load_data(data_file)

    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]

    #log.debug('[MAIN THREAD] train_set len: {}'.format(len(dataset)))
    log.debug('[MAIN THREAD] train_set_x type: {}'.format(type(train_set_x)))
    log.debug('[MAIN THREAD] train_set_x shape: {}'.format(train_set_x.shape))
    train_array = train_set_x.get_value()
    log.debug('[MAIN THREAD] train_array type: {}'.format(type(train_array)))
    log.debug('[MAIN THREAD] train_array shape: {}'.format(train_array.shape))
    log.debug('[MAIN THREAD] train_array[0] type: {}'.format(type(train_array[im_index])))
    log.debug('[MAIN THREAD] train_array[0] shape: {}'.format(train_array[im_index].shape))

    reshaped_im = train_array[im_index].reshape(im_shape)

    log.debug('[MAIN THREAD] reshaped_im type: {}'.format(type(reshaped_im)))
    log.debug('[MAIN THREAD] train_array[0] shape: {}'.format(reshaped_im.shape))

    print_dico = {'test_image': reshaped_im}
    init_dico = {'test_image': reshaped_im.shape}
    print_data = fast_init_image(init_dico)
    print_image_fast(print_dico, print_data)

    import time
    time.sleep(10)

if __name__ == '__main__':
    sgd_optimization_mnist()
    #test()















