##Classification of MNIST Handwritten Digits Database using Deep Learning

This repository contains code meant to classifiy MNIST Handwritten Digits usind Neural Networks. I have used the data from 
http://deeplearning.net/data/mnist/ as well as an algorithmically expanded version of this datset for training the neural 
networks. There are 3 code files which are described below.

####NeuralNetBasic.py
A very simple neural network implemented in python using Stochastic Gradient Descent and Backpropogation. 

####NeuralNetOptimised
An optimised version of neural network above. Incorporates cross-entropy cost function instead of quadratic cost function
L2-regularisation, an algorithmically expanded dataset and better support for performance analysis. The weights have been 
initialised with mean zero and standard deviation 1/sqrt(Number or outputs) rather than 1.

####NeuralNetTheano
Unlike previous two versions, this neural network is implemented using Theano. It also incorporates a couple of convolutional 
layers and a softmax output layer in addition to fully connected layers. Dropout has also been implemented in fully connected 
layers to address the problem of overfitting.


The best efficiency I obtained was 98.87% with a couple of convolutional layers, a fully connected layer of 640 neurons and an 
output softmax layer of 10 neuron, learning rate of 0.1. The network trained over 50 epochs, and took a long while on my MacBook Air (I'm never doing that again).


This code hs been derived from code samples from the book http://neuralnetworksanddeeplearning.com/ authored by Michael Nielson. If you're looking for an introduction to deep learning, the book can be a great starting point. 
