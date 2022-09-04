"""
Author: Sam Armstrong
Date: 2021

Description: Class for applying different activation functions to given data.
"""

import numpy as np

class Activation:
    
    def __init__(self, activation = 'relu') -> None:
        self.activation_function = activation

    # Applies a selected activation function to the given matrix
    def activate(self, x):
        if self.activation_function == 'softmax':
            return self.softmax(x)
        elif self.activation_function == 'relu':
            return self.ReLU(x)
        elif self.activation_function == 'sigmoid':
            return self.logisticSigmoid(x)
        else:
            return x

    # Applies the derivative of a selected activation function to the given matrix
    def activateDerivative(self, x):
        if self.activation_function == 'relu':
            return self.derivativeReLU(x)
        elif self.activation_function == 'sigmoid':
            return self.derivativeSigmoid(x)
        elif self.activation_function == 'softmax':
            return self.derivativeSoftmax(x)
        else:
            return np.ones(x.shape)

    # Softmax activation function (takes a vector rather than a matrix - only applied on the final layer of a network)
    def softmax(self, x):
        ex = np.exp(x - np.max(x))
        y = ex / np.sum(ex, axis = 1)[0]
        return y

    def derivativeSoftmax(self, x):
        sm = self.softmax(x)
        return sm * (1- sm)

    # Applies the ReLU function to a given matrix
    def ReLU(self, h):
        y = h
        if np.any(h < 0) == True:
            y = np.where(h < 0, 0, h)
        return y

    # Applies the derivative of ReLU to a given matrix
    def derivativeReLU(self, h):
        h = np.where(h < 0, 0, h)
        grad_ReLU_matrix = np.where(h > 0, 1, h)
        return grad_ReLU_matrix

    # Applies the sigmoid activation function
    def logisticSigmoid(self, h):
        y = 1 / (1 + np.exp(-1 * h))
        return y

    # Applies the derivative of sigmoid
    def derivativeSigmoid(self, h):
        ls = self.logisticSigmoid(h)
        y = ls * (1 - ls)
        return y
