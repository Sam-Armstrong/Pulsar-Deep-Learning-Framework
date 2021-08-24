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

    def dense(self, Nin, Nout, activation = 'relu'):
        self.layers.append(Dense(Nin, Nout, learning_rate = self.lr, initialization = self.initialization, 
                                 activation = activation, penalty = self.penalty, regularization = self.regularization,
                                 loss = self.loss_function))

    def train(self, training_data, training_labels):
        training_data = NormalizeInput(training_data) # Normalizes the input training data
        Ntrain = len(training_data)
        Nlayers = len(self.layers)

        for n in range(self.epochs):
            batch_number = 1
            dW = 0
            db = 0

            if self.adaptive_lr == True:
                #lr = self.lr / (1 + n) #(1 + (n * self.lr))
                lr = self.lr * np.exp(-0.01 * n)
            else:
                lr = self.lr

            for l in self.layers:
                l.lr = lr

            print(lr)

            shuffled_idxs = np.random.permutation(Ntrain)

            # Loops through all the batches in the training set
            while Ntrain - (batch_number * self.batch_size) >= self.batch_size:
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
                current_batch, gradient, W = layer.backpropagate(x, batch_labels = batch_labels)
                
                i = Nlayers - 2
                # Updates the hidden layers
                while i >= 0:
                    layer = self.layers[i]
                    x = self.getLayersOutput(batch, self.layers[0:i])
                    current_batch, gradient, W = layer.backpropagate(x, next_layer_weights = W, next_layer_grad = gradient)
                    i -= 1
                    
                batch_number += 1

    # Finds the output of a batch after a certain number of layers
    def getLayersOutput(self, batch, layers):
        x = batch
        for l in layers:
            x = l.forwardPass(x)
        return x

    # Allows predictions to be made on a whole batch at a time, speeding up processing
    def batchPredict(self, batch):
        x = batch
        for l in self.layers:
            x = l.forwardPass(x)
        return x
