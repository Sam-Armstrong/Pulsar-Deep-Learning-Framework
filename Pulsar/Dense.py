"""
Author: Sam Armstrong
Date: 2021

Description: Class for creating and interacting with dense (fully connected) network layers.
"""

from Initialization import Initialization
from Activation import Activation
import numpy as np

class Dense:
    
    def __init__(self, Nin, Nout, initialization = 'He', activation = 'relu', learning_rate = 0.01) -> None:
        self.Nin = Nin
        self.Nout = Nout
        self.biases = np.zeros((Nout))
        self.lr = learning_rate
        self.activation = activation

        if initialization == 'He':
            self.weights = Initialization().He_init(Nin, Nout)
        elif initialization == 'Xavier':
            self.weights = Initialization().Xavier_init(Nin, Nout)
        else:
            self.weights = Initialization().Random_init(Nin, Nout)

    # Calculates the layer output for a given batch
    def forwardPass(self, batch):
        a = Activation(self.activation)
        h = np.transpose(np.matmul(self.weights, np.transpose(batch)))
        h = np.add(h, self.biases[np.newaxis,:]) # Adds the biases
        y = a.activate(h) # Applies the activation function
        return y

    # Trains the layer for a batch through backpropagation
    def backpropagate(self, batch, batch_labels = None, next_layer_weights = None, next_layer_grad = None):
        a = Activation(self.activation)
        h = np.transpose(np.matmul(self.weights, np.transpose(batch)))
        h = np.add(h, self.biases[np.newaxis,:]) # Adds the biases
        y = a.activate(h) # Applies the activation function
        derivative_matrix = a.activateDerivative(h) # Applies the derivative of the activation function

        # Calculates the local gradients for each of the neurons in the layer
        try:
            delta = np.multiply(np.transpose(np.matmul(np.transpose(next_layer_weights), np.transpose(next_layer_grad))), derivative_matrix)
        except:
            # Calculates the error signal
            e_n = np.subtract(batch_labels, y)
            # Calculates the local gradient of each neuron
            delta = np.multiply(e_n, derivative_matrix)

        # Updates the weights and biases
        self.weights += (np.matmul(delta.T, batch) * self.lr) / len(batch)
        self.biases += (sum(delta) * self.lr) / len(batch)
        
        return y, delta, self.weights
