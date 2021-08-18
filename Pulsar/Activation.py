"""
Author: Sam Armstrong
Date: 2021

Description: Class for applying different activation functions to given data.
"""

import numpy as np

class Activation:
    
    def __init__(self) -> None:
        pass

    # Softmax activation function (takes a vector rather than a matrix - only applied on the final layer of a network)
    def softmax(self, x):
        ex = np.exp(x)
        y = ex / np.sum(ex)
        return y

    # Applies the ReLU function to a given matrix
    def ReLU(self, h):
        y = h
        
        if np.any(h < 0) == True:
            y = np.where(h < 0, 0, h)
            
        return y

    # Applies the derivation of ReLU to a given matrix
    def derivationReLU(self, h):
        h = np.where(h < 0, 0, h)
        grad_ReLU_matrix = np.where(h > 0, 1, h)

        return grad_ReLU_matrix

    # Applies the sigmoid activation function
    def logisticSigmoid(self, h):
        y = 1 / (1 + np.exp(-1 * h))
        return y

    # Applies the derivation of sigmoid
    def derivationSigmoid(self, h):
        ls = self.logisticSigmoid(h)
        y = ls * (1 - ls)
        return y
