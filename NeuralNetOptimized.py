import gzip 
import cPickle
import os 
import numpy as np
import random
import time
import json

def load_data():
    """
    Loads data from file. Returns a tuple of 3 lists, containing training data,
    validation data and test data in order. 
    
    The training data , validation and test data are tuples of two numpy arrays 
    of length 10,000 each. First of these is contains 784x1 numpy arrays which 
    represents the pixel intensities of the image. The second contains integers 
    representing the correct  classification for examples of the corresponding
    indexes.
    """
    
    f = gzip.open('./mnist_expanded.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    return training_data, validation_data, test_data

def transform_data():
    """
    Tranform the data into a format which is more feasible for training.
    
    Returns a a 3-tuple of containing training data validation data and test
    data in order. 
    
    The training data is now an list of 50,000 tuples representing each training 
    example. Each tuple consists of a 784x1 numpy array, representing pixel 
    intensities and a 10x1 numpy array, with 0 for all indexes but 1 for theindex 
    corresponding to the correct classification of the example image.
    
    The training data is now an list of 50,000 tuples representing each training 
    example. Each tuple consists of a 784x1 numpy array, representing pixel 
    intensities and an integer  corresponding to the correct classification of
    the image example.
    """
    data = load_data()
    td, vd, ttd = data[0], data[1], data[2]
    X_train = [np.reshape(x, (784,1)) for x in td[0]]
    Y_train = [vectorize(y) for y in td[1]]
    train_data = zip(X_train, Y_train)
    X_val = [np.reshape(x, (784,1)) for x in vd[0]]
    X_test = [np.reshape(x, (784,1)) for x in ttd[0]]
    val_data = zip(X_val, vd[1])
    test_data = zip(X_test, ttd[1])
    return train_data, val_data, test_data
    
    
  
def vectorize(s):
    """
    Returns a 10x1 numpy array with all indices 0 except for sth indice
    """
    result = np.zeros((10,1))
    result[s] = 1
    return result
    
def vectorize_matrix(yv):
    
    """
    Takes in an array, yv of ints.
    
    Returns a matrix such that element ij, where i is the value at jth element
    in array yv (i = yv[j]), is 1. All the other elements are 0.
    """
    
    temp = np.zeros((10, yv.shape[0]))
    for example in range(yv.shape[0]):
        temp[yv[example]][example] = 1
    return temp
    
   
class QuadCost(object):
    @staticmethod
    def fn(a,y):
        return sum(0.5*np.linalg.norm(a-y, axis = 0)**2)
        
    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)

        
class CECost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""

        return (a-y)


class NeuralNet(object):
    def __init__(self, sizes, cost = CECost, large_weights = False):
        self.sizes = sizes
        self.cost = cost
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        if large_weights:
            self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]
        else:
            self.weights = [np.random.randn(x,y)/np.sqrt(x) for x,y in zip(sizes[1:], sizes[:-1])]
    
    def feedforward(self, inp):
        a = inp
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b) 
        return a
        
 
    def SGD(self, td, epochs, mbs, eta, lmbda = 0.0, test_data = None, train_data = None, test_accuracy =  True, test_cost = True, train_accuracy = True, train_cost = True):
        """
        Stochastic Gradient Descent.
        
        td:         training data to perform SGD upon.
        epochs:     Number of epochs or full iterations over the dataset.
        mbs:        Size of mini-batch used.
        eta:        Learning Rate
        test_data:  If test data is present, the function tests the model over
                    test data, and returns the accuracy.
        test_accuracy: prints accuracy of thr NN on test data
        train_accuracy: prints accuracy of the model on training data
        test_cost:  prints the cost of NN on test data
        train_cost: prints cost of NN of train data
        
        """
        test_accuracies, test_costs, train_accuracies, train_costs = [], [], [], []   
        
        for x in xrange(epochs):
            
            mini_batches = []
            random.shuffle(td)
            
            for i in range(0, len(td), mbs):
                mini_batches.append(np.array(td[i:i+mbs]))
                
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(td))
            
            if test_data:
                print("Epoch :",x,"Completed")
                
            
            if test_accuracy:
                accuracy = self.evaluate(test_data)
                print "Test Accuracy for this epoch is",round(accuracy,4)
                test_accuracies.append(accuracy)
                
            if test_cost:   
                cost = self.total_cost(test_data, lmbda, convert = True)
                test_costs.append(cost)
                print "Test Cost for this epoch is", round(cost, 3)
            
            if train_accuracy:
                accuracy = self.evaluate(train_data, convert = True)
                train_accuracies.append(accuracy)
                print "The training accuracy is", round(accuracy, 3)
            
            if train_cost:   
                cost = self.total_cost(train_data, lmbda)
                train_costs.append(cost)
                print "Training Cost for this epoch is", round(cost, 3)
            
            print
                    
            
            
            
                
             
    def update_mini_batch(self, mini_batches, eta, lmbda, n):
        
        """
        Updates the parameters of the model using the backpropogation algorithm
        over all examples.
        
        mini_batches: array of mini_batches
        eta         : Learning Rate
        lmbda:      : L2-Regularisation constant
        n:          : Number of mini_batches
        """
        nabla_w  = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        xv = np.asarray([x.ravel() for (x,y) in mini_batches]).transpose()
        yv = np.asarray([y.ravel() for (x,y) in mini_batches]).transpose()
        
        delta_b, delta_w = self.backprop(xv,yv)
    
        
        nabla_w = [nw + ndw for  nw, ndw in zip(nabla_w, delta_w)]
        nabla_b = [nb + ndb for  nb, ndb in zip(nabla_b, delta_b)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batches))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases =  [b-(eta/n)*nb for b, nb in zip(self.biases, nabla_b)]
            
    def backprop(self,x,y):
        """
        Backpropogation Algorithm. Calculates the gradient for the entire set
        of paramters of a model given a training example and it's output using
        Backpropogation.
        """
        nabla_w  = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
    
        delta = (self.cost).delta(zs[-1], activations[-1], y)    
        nabla_b[-1] = delta.sum(1).reshape(len(delta), 1)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for i in xrange(2, self.num_layers):
            
            delta = (np.dot(self.weights[-i + 1].transpose(), delta))*sigmoid_prime(zs[-i])
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
            nabla_b[-i] =  delta.sum(1).reshape(len(delta), 1)
        
        return nabla_b, nabla_w

        
        
    def evaluate(self, test_data, convert = False):
        """
        Evaluates the performance of the neural over test data.
        Returns classification accuracy.
        
        """
        xv = np.asarray([x.ravel() for (x,y) in test_data]).transpose()
        
        if convert:
            yv = np.asarray([np.argmax(y) for (x,y) in test_data]).transpose()
        else:
            yv = np.asarray([y for (x,y) in test_data]).transpose()
        
        result = np.argmax(self.feedforward(xv), axis = 0)

        return sum(yv == result)/float(yv.shape[0])
   
    
        
    def cost_derivative(self, output_activations, y):
           """Return the vector of partial derivatives \partial C_x /
           \partial a for the output activations."""
           return (output_activations-y)
    
    def total_cost(self, data, lmbda, convert = False):
        """
        Returns the cost of Neural Network on data. 
        
        lamba: L2-Regularisation constant
        """
        cost = 0.0
        xv = np.asarray([x.ravel() for (x,y) in data]).transpose()
        yv = np.asarray([y for (x,y) in data]).transpose()
        
        a = self.feedforward(xv)
        
        if convert:
            yv = vectorize_matrix(yv)   
        
        cost = self.cost.fn(a, yv)/len(data)
        
        
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        
        return cost
    

        return cost
            
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
               
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
     
        
data = transform_data()


training_data, validation_data, test_data = data[0], data[1], data[2]

net = NeuralNet([784, 30,  10])

a = time.time()
net.SGD(training_data, 30, 10, 3.0, test_data = test_data, train_data=training_data)
#net.SGDa(training_data, 30, 10, 3.0, eval_data = test_data)

b = time.time()
print "The time taken for learning is", b-a