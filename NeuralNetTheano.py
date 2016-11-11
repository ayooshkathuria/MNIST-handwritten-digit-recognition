# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 00:54:04 2016

@author: ayooshmac
"""


import numpy as np 
import cPickle
import gzip

import theano
import theano.tensor as T
from theano.tensor.nnet import conv 
from theano.tensor.signal import pool
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
#Activation functions for neurons
def linear(z): 
    return z
    
def ReLu(x):
    return T.max(0, x)
    
from theano.tensor.nnet import sigmoid

GPU = True
if GPU:
    print "Trying to run under a GPU.  If this is not wanted, then modify "+\
        "set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not wanted, then the modify "+\
        "network3.py to set\nthe GPU flag to True."


def load_data_shared(filename="./mnist.pkl.gz"):
    """
    Loads data from file. Returns a tuple of 3 lists, containing training data,
    validation data and test data in order. 
    
    The training data , validation and test data are tuples of two numpy arrays 
    of length 10,000 each. First of these is contains 784x1 numpy arrays which 
    represents the pixel intensities of the image. The second contains integers 
    representing the correct  classification for examples of the corresponding
    indexes.
    """
    
    f = gzip.open(filename, "rb")
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        """
        Place the data into shared variables. This allows Theano to put the data in GPU,
        if one is availible
        """
        
        x = theano.shared(np.asarray(data[0], dtype = theano.config.floatX), borrow = True)
        y = theano.shared(np.asarray(data[1], dtype = theano.config.floatX), borrow = True)
        return x, T.cast(y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]


training_data, validation_data, test_data = load_data_shared()

class ConvPoolLayer(object):
    def __init__(self, filter_shape, image_shape, pool_size = (2,2), activation_fn = sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.activation_fn = activation_fn
        
        #Inititialising weights 
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(pool_size))
        self.w = theano.shared(np.asarray(np.random.normal(loc = 0.0, scale =np.sqrt(1.0/n_out), size = filter_shape),
                                          dtype = theano.config.floatX), 
                                          borrow = True)
        self.b = theano.shared(np.asarray(np.random.normal(loc = 0.0, scale = 1.0,
                                                           size = (filter_shape[0],)), dtype=theano.config.floatX), borrow = True)
        
        self.params = [self.w, self.b]
        
    
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(self.inpt, filters = self.w, filter_shape=self.filter_shape,
                               image_shape=self.image_shape)
        pooled_out = pool.pool_2d(input=conv_out, ds = self.pool_size, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output

class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn = sigmoid, p_dropout = 0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        
        #Initiaze the weights 
        self.w = theano.shared(np.asarray(np.random.normal(loc = 0.0, scale = 1.0/n_out, size = (n_in, n_out))
         ,dtype = theano.config.floatX), borrow = True, name="w")
        

        self.b = theano.shared(np.asarray(np.random.normal(loc = 0.0, scale = 1.0, size = (n_out,)),
                                          dtype = theano.config.floatX), borrow = True, name="b")
        
        
        self.params = [self.w, self.b]
    
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn((1- self.p_dropout)*(T.dot(self.inpt, self.w) + self.b))
        self.y_out = T.argmax(self.output, axis = 1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)
    
    def accuracy(self,y):
        return T.mean(T.eq(y, self.y_out))
        
    
        
        
class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, p_dropout = 0.0):
        self.n_in = n_in
        self.n_out = n_out 
        self.p_dropout= p_dropout
        
        #Initialize the weights 
        self.w = theano.shared(np.asarray(np.zeros((n_in, n_out)),dtype = theano.config.floatX),
                               borrow = True, name = "w")
        self.b = theano.shared(np.asarray(np.zeros((n_out,)), dtype= theano.config.floatX), borrow = True, name = "b")
     
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)
        
    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


def size(data):
    """Returns the size of the data"""
    return data[0].get_value(borrow = True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    mask = srng.binomial(size = layer.shape, n = 1, p = 1 - p_dropout)
    return layer*T.cast(mask, theano.config.floatX)

class Network(object):

    def __init__(self, layers, mini_batch_size):
        self.layers = layers 
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        init_layer = self.layers[0]
        print init_layer
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda = 0.0):
        training_x, training_y = training_data[0], training_data[1]
        validation_x, validation_y = validation_data[0], validation_data[1]
        test_x, test_y = test_data[0], test_data[1]  
        
        #calculate the size of batches
        training_batches = size(training_data)/(mini_batch_size)
        validation_batches = size(validation_data)/mini_batch_size
        test_batches = size(test_data)/mini_batch_size
        
        #define the symbolic realtion of l2 norm. 
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + (0.5)*(lmbda)*(l2_norm_squared)/(training_batches)
        grads = T.grad(cost,self.params)
        updates = [(param, param - eta*grad) for param, grad in zip(self.params, grads)]
        i = T.lscalar()
        
#        
        train_mb =  theano.function([i], cost, updates = updates, 
                                    givens = {self.x: training_x[i*self.mini_batch_size : (i+1)*self.mini_batch_size], self.y: training_y[i*self.mini_batch_size : (i+1)*self.mini_batch_size]})
        
        validate_mb_accuracy =theano.function([i], self.layers[-1].accuracy(self.y), 
                                              givens = {self.x: validation_x[i*self.mini_batch_size : (i+1)*self.mini_batch_size], 
                                                                 self.y: validation_y[i*self.mini_batch_size : (i+1)*self.mini_batch_size]})
        theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
            
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            }) 

        #Do the actual training data
        best_val_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in range(training_batches):
                iteration = epoch*(training_batches) + minibatch_index
                if (iteration)%1000 == 0:
                    print("Batch Number", iteration, "trained")
                train_mb(minibatch_index)
                if (iteration + 1)%(training_batches) == 0:
                  
                    validation_accuracy = np.mean([validate_mb_accuracy(j) for j in xrange(validation_batches)])
                    print "Epoch", epoch, "completed: ", "Validation accuracy:", validation_accuracy
                    
                    if validation_accuracy>= best_val_accuracy:
                        print "This is the best validation accuracy so far"
                        if test_data:
                            test_accuracy= np.mean([validate_mb_accuracy(j) for j in xrange(test_batches)])
                            print "The test accuracy for this epoch is :", round(test_accuracy,4)
                            
                                                                    
training_data, validation_data, test_data = load_data_shared()
mini_batch_size = 10

        
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                      filter_shape=(20, 1, 5, 5), 
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                      filter_shape=(40, 20, 5, 5), 
                      pool_size=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, 
            validation_data, test_data)    
