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
        #self.biases = np.random.uniform(0, 1, Nin)
        self.lr = learning_rate

        if initialization == 'He':
            self.weights = Initialization().He_init(Nin, Nout)
        elif initialization == 'Xavier':
            self.weights = Initialization().Xavier_init(Nin, Nout)
        else:
            self.weights = Initialization().Random_init(Nin, Nout)

    def forwardPass(self, batch):
        a = Activation()

        # Calculates the layer output prior to applying the activation function
        h = np.transpose(np.matmul(self.weights, np.transpose(batch)))
        h = np.add(h, self.biases[np.newaxis,:]) # Adds the biases
        
        # TODO the selected activation function needs to be used here
        y = a.ReLU(h) # Applies the activation function

        return y

    def backpropagate(self, batch, batch_labels = None, next_layer_weights = None, next_layer_grad = None):
        a = Activation()

        # Calculates the layer output prior to applying the activation function
        h = np.transpose(np.matmul(self.weights, np.transpose(batch)))
        h = np.add(h, self.biases[np.newaxis,:]) # Adds the biases
        
        # TODO the selected activation function needs to be used here
        y = a.ReLU(h) # Applies the activation function

        derivation_matrix = a.derivationReLU(h)

        try:
            delta = np.multiply(np.transpose(np.matmul(np.transpose(next_layer_weights), np.transpose(next_layer_grad))), derivation_matrix)
        except:
            # Calculates the error signal
            e_n = np.subtract(batch_labels, y)
            # Calculates the local gradient of each neuron
            delta = np.multiply(e_n, derivation_matrix)

        self.weights += (np.matmul(delta.T, batch) * self.lr) / len(batch)
        self.biases += (sum(delta) * self.lr) / len(batch)
        
        return y, delta, self.weights

#print(Dense(3,3).forwardPass(np.array([[-1, 2, 3], [4, -5, 6]], np.int32)))