"""
Author: Sam Armstrong
Date: 2021

Description: Class for creating and interacting with dense (fully connected) network layers.
"""

from Initialization import Initialization
from Activation import Activation
import numpy as np

class Dense:
    
    def __init__(self, Nin, Nout, initialization = 'He', activation = 'relu') -> None:
        self.Nin = Nin
        self.Nout = Nout
        self.biases = np.zeros((Nout))

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
        h += self.biases[np.newaxis,:] # Adds the biases
        
        # TODO the selected activation function needs to be used here
        y = a.ReLU(h) # Applies the activation function

        return y
