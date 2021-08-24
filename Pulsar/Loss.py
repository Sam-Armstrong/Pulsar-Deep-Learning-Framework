"""
Author: Sam Armstrong
Date: 2021

Description: Class that contains different loss functions and their derivatives.
"""

import numpy as np
from Activation import Activation

class Loss:

    def __init__(self, loss_function) -> None:
        self.loss = loss_function

    def derivativeLoss(self, batch_labels, y, derivative_matrix):
        if self.loss == 'cross-entropy':
            delta = self.derivativeCrossEntropy(batch_labels, y)
        else:
            delta = self.derivativeMeanSquare(batch_labels, y, derivative_matrix)

        return delta
    
    def derivativeMeanSquare(self, batch_labels, y, derivative_matrix):
        e_n = np.subtract(batch_labels, y)
        delta = np.multiply(e_n, derivative_matrix)
        return delta

    def derivativeCrossEntropy(self, batch_labels, y):
        delta = np.subtract(batch_labels, y)
        return delta
        