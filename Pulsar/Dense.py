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
    
    def __init__(self, Nin, Nout, initialization = 'He', activation = 'relu', learning_rate = 0.01, 
                 regularization = 'L2', penalty = 0, loss = 'cross-entropy', batch_size = 200, optimizer = 'SGD') -> None:
        self.Nin = Nin
        self.Nout = Nout
        self.biases = np.zeros((Nout))
        self.lr = learning_rate
        self.activation = activation
        self.reg = regularization
        self.penalty = penalty
        self.loss = loss
        self.batch_size = batch_size
        self.a = Activation(self.activation)
        self.optimizer = optimizer

        self.m = 0
        self.s = 0
        self.m_bias = 0
        self.s_bias = 0

        if initialization == 'He':
            self.weights = Initialization().He_init(Nin, Nout)
        elif initialization == 'Xavier':
            self.weights = Initialization().Xavier_init(Nin, Nout)
        else:
            self.weights = Initialization().Random_init(Nin, Nout)

    # Calculates the layer output for a given batch
    def forwardPass(self, batch):
        batch = batch.reshape(len(batch), self.Nin)
        h = np.transpose(np.matmul(self.weights, np.transpose(batch)))
        h = np.add(h, self.biases[np.newaxis,:]) # Adds the biases
        y = self.a.activate(h) # Applies the activation function
        return y

    # Trains the layer for a batch through backpropagation
    def backpropagate(self, batch, batch_labels = None, next_layer_weights = None, next_layer_grad = None):
        batch = batch.reshape(self.batch_size, self.Nin)
        h = np.transpose(np.matmul(self.weights, np.transpose(batch)))
        h = np.add(h, self.biases[np.newaxis,:]) # Adds the biases
        y = self.a.activate(h) # Applies the activation function
        derivative_matrix = self.a.activateDerivative(h) # Applies the derivative of the activation function

        # Calculates the local gradients for each of the neurons in the layer
        if next_layer_weights is None and next_layer_grad is None and batch_labels is not None: # If this is the output layer
            l = Loss(self.loss)
            delta = l.derivativeLoss(batch_labels, y, derivative_matrix)
        elif batch_labels is None: # If this is a hidden layer
            delta = np.multiply(np.transpose(np.matmul(np.transpose(next_layer_weights), np.transpose(next_layer_grad))), derivative_matrix)
        else:
            delta = next_layer_grad.reshape(self.batch_size, self.Nout)

        # Updates the weights and biases
        if self.optimizer == 'sgd':
            delta_batch = np.matmul(delta.T, batch)
            self.weights += (delta_batch * self.lr) / len(batch)
            self.biases += (sum(delta) * self.lr) / len(batch)
        
        elif self.optimizer == 'adam':
            # Adam parameters are currently just set to default values
            beta1 = 0.9
            beta2 = 0.99999
            epsilon = 0.0001

            delta_batch = np.matmul(delta.T, batch) # Derivative of the Error function with respect to the weights
            m = (beta1 * self.m) - ((1 - beta1) * delta_batch)
            s = (beta2 * self.s) + ((1 - beta2) * (delta_batch ** 2))
            m_hat = m / (1 - beta1)
            s_hat = s / (1 - beta2)
            self.m = m
            self.s = s

            m_bias = (beta1 * self.m_bias) - ((1 - beta1) * sum(delta))
            s_bias = (beta2 * self.s_bias) + ((1 - beta2) * (sum(delta) ** 2))
            m_bias_hat = m_bias / (1 - beta1)
            s_bias_hat = s_bias / (1 - beta2)
            self.m_bias = m_bias
            self.s_bias = s_bias

            self.weights -= self.lr * (m_hat / np.sqrt(s_hat + epsilon))
            self.biases -= self.lr * m_bias_hat / np.sqrt(s_bias_hat + epsilon)


        # Applies the chosen regularization method
        if self.penalty != 0:
            if self.reg == 'L2':
                self.weights -= L2_penalty_matrix(self.weights, self.penalty)
                self.biases -= L2_penalty_matrix(self.biases, self.penalty)
            else:
                self.weights -= L1_penalty_matrix(self.weights, self.penalty)
                self.biases -= L1_penalty_matrix(self.biases, self.penalty)
        
        return delta, self.weights
