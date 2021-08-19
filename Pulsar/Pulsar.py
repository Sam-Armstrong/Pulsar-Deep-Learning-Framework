"""
Author: Sam Armstrong
Date: 2021

Description: The main class that is used for implementing neural networks.
"""

from Dense import Dense
import numpy as np

# Normalizes a set of data
def NormalizeInput(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

class Pulsar:
    
    def __init__(self, optimizer = 'sgd', learning_rate = 0.01, batch_size = 200, epochs = 100, 
                 regularization = 'l1', loss = 'mean-squared') -> None:
        self.optimizer = optimizer
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.regularization = regularization
        self.loss_function = loss
        self.layers = list()

    def dense(self, Nin, Nout):
        self.layers.append(Dense(Nin, Nout))

    def train(self, training_data, training_labels):
        training_data = NormalizeInput(training_data) # Normalizes the input training data
        Ntrain = len(training_data)
        Nlayers = len(self.layers)

        for n in range(self.epochs):
            batch_number = 1
            dW = 0
            db = 0

            shuffled_idxs = np.random.permutation(Ntrain)

            # Loops through all the batches in the training set
            while Ntrain - (batch_number * self.batch_size) >= self.batch_size:
                start_index = batch_number * self.batch_size
                batch = np.zeros((self.batch_size, 784))
                batch_labels = np.zeros((self.batch_size, 26))

                idxs = shuffled_idxs[start_index:(start_index + self.batch_size)]

                # Collects the next batch
                for i in range(len(idxs)):
                    batch[i] = training_data[idxs[i]]
                    batch_labels[i] = training_labels[idxs[i]]

                layer = self.layers[Nlayers - 1]
                current_batch, gradient, W = layer.backpropagate(batch, batch_labels = batch_labels)
                
                for i in range(Nlayers):
                    layer = self.layers[Nlayers - (i + 1)]
                    current_batch, gradient, W = layer.backpropagate(current_batch, next_layer_weights = W, next_layer_grad = gradient)

                batch_number += 1
