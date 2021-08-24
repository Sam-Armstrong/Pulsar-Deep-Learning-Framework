"""
Author: Sam Armstrong
Date: 2021

Description: Class for creating and interacting with dense (fully connected) network layers.
"""

from Initialization import Initialization
from Activation import Activation
from Loss import Loss
import numpy as np

# Returns a matrix of positive and negative penalty values corresponding to whether each weight is positive or negative
def L1_penalty_matrix(W, penalty):
    W = np.where(W < 0, -1 * penalty, W)
    W = np.where(W > 0, penalty, W)
    return W

def L2_penalty_matrix(W, penalty):
    W = np.where(W != 0, penalty * W, W)
    return W

class Dense:
    
    def __init__(self, Nin, Nout, initialization = 'He', activation = 'relu', learning_rate = 0.01, regularization = 'L2', penalty = 0, loss = 'cross-entropy') -> None:
        self.Nin = Nin
        self.Nout = Nout
        self.biases = np.zeros((Nout))
        self.lr = learning_rate
        self.activation = activation
        self.reg = regularization
        self.penalty = penalty
        self.loss = loss

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
            # Finds the gradient of the selected loss function
            l = Loss(self.loss)
            delta = l.derivativeLoss(batch_labels, y, derivative_matrix)

        # Updates the weights and biases
        self.weights += (np.matmul(delta.T, batch) * self.lr) / len(batch)
        self.biases += (sum(delta) * self.lr) / len(batch)

        if self.penalty != 0:
            if self.reg == 'L2':
                self.weights += -1 * L2_penalty_matrix(self.weights, self.penalty)
            else:
                self.weights += -1 * L1_penalty_matrix(self.weights, self.penalty)
        
        return y, delta, self.weights
