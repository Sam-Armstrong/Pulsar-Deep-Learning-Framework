"""
Author: Sam Armstrong
Date: 2021

Description: The main class that is used for implementing neural networks.
"""

from Dense import Dense
from Convolution import Convolution
from Pooling import Pooling
import numpy as np

# Normalizes a set of data
def NormalizeInput(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

class Pulsar:
    
    def __init__(self, optimizer = 'sgd', learning_rate = 0.01, batch_size = 200, epochs = 5, 
                 regularization = 'L2', loss = 'cross-entropy', initialization = 'He', penalty = 0, 
                 adaptive_lr = True) -> None:
        self.optimizer = optimizer
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.regularization = regularization
        self.loss_function = loss
        self.initialization = initialization
        self.penalty = penalty
        self.adaptive_lr = adaptive_lr
        self.layers = list()

    # Adds a fully-connected layer to the network
    def dense(self, Nin, Nout, activation = 'relu'):
        self.layers.append(Dense(Nin, Nout, learning_rate = self.lr, initialization = self.initialization, 
                                 activation = activation, penalty = self.penalty, regularization = self.regularization,
                                 loss = self.loss_function, batch_size = self.batch_size, optimizer = self.optimizer))

    # Adds a convolutional layer to the network
    def convolution(self, input_height, input_width, kernel_size = 3, depth = 1, input_depth = 1, padding = 0, stride = 1):
        self.layers.append(Convolution(input_height, input_width, kernel_size = kernel_size, depth = depth, 
                                       input_depth = input_depth, batch_size = self.batch_size, learning_rate = self.lr, 
                                       padding = padding, stride = stride, optimizer = self.optimizer))

    # Adds a pooling layer to the network
    def pooling(self, input_height, input_width, mode = 'max', spatial_extent = 2, stride = 2, depth = 1):
        self.layers.append(Pooling(input_height, input_width, mode = mode, spatial_extent = spatial_extent, 
                                   stride = stride, batch_size = self.batch_size, depth = depth))

    # Trains the defined network
    def train(self, training_data, training_labels):
        training_data = NormalizeInput(training_data) # Normalizes the input training data
        Ntrain = len(training_data)
        Nlayers = len(self.layers)

        for n in range(self.epochs):
            print('Epoch %s' % (n + 1))
            batch_number = 1

            shuffled_idxs = np.random.permutation(Ntrain)

            # Loops through all the batches in the training set
            while Ntrain - (batch_number * self.batch_size) >= self.batch_size:
                print(batch_number)
                start_index = batch_number * self.batch_size
                batch = np.zeros((self.batch_size, 784))
                batch_labels = np.zeros((self.batch_size, 10))

                idxs = shuffled_idxs[start_index:(start_index + self.batch_size)]

                # Collects the next batch
                for i in range(len(idxs)):
                    batch[i] = training_data[idxs[i]]
                    batch_labels[i] = training_labels[idxs[i]]

                # Updates the final layer
                layer = self.layers[Nlayers - 1]
                x = self.getLayersOutput(batch, self.layers[0:Nlayers - 1])
                #if type(layer) == Dense:
                gradient, W = layer.backward(x, batch_labels = batch_labels)
                
                i = Nlayers - 2
                # Updates the hidden layers
                while i >= 0:
                    layer = self.layers[i]
                    x = self.getLayersOutput(batch, self.layers[0:i])
                    if type(layer) == Dense:
                        gradient, W = layer.backward(x, next_layer_weights = W, next_layer_grad = gradient)
                    elif type(layer) == Convolution:
                        gradient = layer.backward(x, next_layer_weights = W, next_layer_grad = gradient)
                    else:
                        gradient = layer.backward(x, next_layer_weights = W, next_layer_grad = gradient)
                    i -= 1
                    
                batch_number += 1

    # Finds the output of a batch after a certain number of layers
    def getLayersOutput(self, batch, layers):
        x = batch
        for l in layers:
            x = l.forward(x)
        return x

    # Allows predictions to be made on a whole batch at a time, speeding up processing
    def batchPredict(self, batch):
        x = batch
        for l in self.layers:
            x = l.forward(x)
        return x
