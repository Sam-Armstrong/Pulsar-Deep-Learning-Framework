"""
Author: Sam Armstrong
Date: 2021

Description: Class for creating and interacting with convolutional network layers for automatic feature extraction.
"""

from Activation import Activation
import numpy as np
from scipy import signal

class Convolution:
    
    def __init__(self, input_height, input_width, kernel_size = 3, depth = 1, input_depth = 1, batch_size = 200,
                 stride = 1, padding = 0, activation = 'relu', learning_rate = 0.001, optimizer = 'sgd') -> None:
        # Attributes of this convolution object
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.a = Activation(activation = activation)
        self.optimizer = optimizer
        self.stride = stride
        self.padding = padding
        self.depth = depth # The number of kernels (depth of the output)
        self.input_depth = input_depth # The depth of the input
        self.input_height = input_height
        self.input_width = input_width
        self.output_shape = (batch_size, depth, ((input_height + (padding * 2)) - kernel_size + stride) / stride, ((input_width + (padding * 2)) - kernel_size + stride) / stride) # The shape of the output from the layer
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size) # The shape of the kernels (filters)
        
        self.m = 0
        self.s = 0
        self.m_bias = 0
        self.s_bias = 0

        # Initialize the filters and biases
        self.kernels = np.random.randn(*self.kernels_shape) * (1 / (kernel_size * kernel_size * input_depth))
        self.biases = np.zeros((depth, int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride), int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride)))


    def forward(self, batch):
        """if len(np.shape(batch)) == 4:
            current_batch_size = len(batch)
        else:
            current_batch_size = self.batch_size"""
        current_batch_size = len(batch)

        # Reshapes the data
        batch = batch.reshape(current_batch_size, self.input_depth, self.input_height, self.input_width)
        output_batch = np.empty((current_batch_size, self.depth, int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride), int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride)))

        # Loop through the number of output kernels to form an output feature map with the correct depth (number of output channels)
        for n in range(current_batch_size):
            current_output_batch = np.empty((self.depth, int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride), int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride)))
            data = batch[n]

            # Applies the chosen padding
            if self.padding != 0:
                data = np.pad(data, self.padding)
                for i in range(self.padding):
                    # Removes unnecessary third-dimension padding
                    data = np.delete(data, 0, axis = 0)
                    data = np.delete(data, len(data) - 1, axis = 0)

            # Convolves the input feature map with each filter
            for i in range(self.depth):
                output = np.copy(self.biases[i]) # The ouput is the bias + the convolution output

                filter = self.kernels[i]
                output += signal.correlate(data, filter, mode = 'valid')[0] # Performs cross-correlation (equivalent to convolution)

                current_output_batch[i] = output

            output_batch[n] = current_output_batch

        self.h = output_batch # Output prior to activation
        output_batch = self.a.activate(output_batch) # Applies the activation function
        
        return output_batch


    def backward(self, batch, batch_labels = None, next_layer_weights = None, next_layer_grad = None):

        current_batch_size = len(batch)

        if len(next_layer_grad.shape) == 4:
            next_layer_grad = next_layer_grad.reshape(current_batch_size, next_layer_grad.shape[3] * next_layer_grad.shape[2] * next_layer_grad.shape[1])

        # Finds the output gradient if the next layer is dense - as dense layers do not return their input gradient
        try:
            fp = self.forward(batch).reshape(current_batch_size, int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride) * int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride) * self.depth)
            derivative_matrix = Activation().derivativeReLU(fp)
            output_gradient = np.multiply(np.transpose(np.matmul(np.transpose(next_layer_weights), np.transpose(next_layer_grad))), derivative_matrix)
        except Exception as e:
            output_gradient = next_layer_grad

        # dL/dF = Convolution of the input and the loss gradient
        # dL/dX = Full Convolution of the 180 degree rotated filter and the loss gradient
        
        if len(output_gradient.shape) == 2:
            output_gradient_size = self.input_width + (self.padding * 2) - self.kernel_size + 1
            output_gradient = output_gradient.reshape(self.batch_size, self.depth, int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride), int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride))###
        else:
            output_gradient_size = output_gradient.shape[len(output_gradient - 1)]

        batch = batch.reshape(self.batch_size, self.input_depth, self.input_height, self.input_width)
        filter_gradient = np.zeros((self.depth, self.input_depth, self.kernel_size, self.kernel_size))
        input_gradient = np.empty((self.batch_size, self.input_depth, self.input_height + self.padding * 2, self.input_width + self.padding * 2))

        max_padding = output_gradient_size - 1 # The largest padding that can be applied

        for n in range(self.batch_size):
            current_filter_batch = np.empty((self.depth, self.input_depth, int((self.input_height - output_gradient_size + (self.padding * 2) + self.stride) / self.stride), int((self.input_width - output_gradient_size + (self.padding * 2) + self.stride) / self.stride)))
            input_batch_grad = np.empty((self.input_depth, self.input_height + self.padding * 2, self.input_width + self.padding * 2)) # + self.padding * 2
            data = batch[n]

            # Applies the chosen padding
            old_data = data
            if self.padding != 0:
                data = np.pad(data, self.padding)
                for i in range(self.padding):
                    data = np.delete(data, 0, axis = 0)
                    data = np.delete(data, len(data) - 1, axis = 0)

            for i in range(self.depth):  
                filter_x_positions = int((self.input_width - output_gradient_size + (self.padding * 2) + self.stride) / self.stride)
                filter_y_positions = int((self.input_height - output_gradient_size + (self.padding * 2) + self.stride) / self.stride)
                filter_output = np.zeros((self.input_depth, filter_y_positions, filter_x_positions))

                input_x_positions = int(((self.kernel_size + (max_padding * 2)) - output_gradient_size + self.stride) / self.stride)
                input_y_positions = int(((self.kernel_size + (max_padding * 2)) - output_gradient_size + self.stride) / self.stride)
                input_grad = np.zeros((input_y_positions, input_x_positions))
                
                for d in range(self.input_depth):
                    filter = self.kernels[i][d]
                    filter_output[d] = signal.correlate(data[d], output_gradient[n][i], mode = 'valid').reshape(filter_y_positions, filter_x_positions)
                    input_batch_grad[d] += signal.convolve(output_gradient[n][i], filter, mode = 'full')

                current_filter_batch[i] = filter_output

            filter_gradient += current_filter_batch
            input_gradient[n] = input_batch_grad # / depth?

        if self.optimizer == 'sgd':
            self.kernels -= (self.learning_rate * filter_gradient) / current_batch_size
            self.biases -= (self.learning_rate * np.sum(output_gradient, axis = 0)) / current_batch_size
        
        elif self.optimizer == 'adam':
            # Adam parameters are currently just set to default values
            beta1 = 0.9
            beta2 = 0.99999
            epsilon = 0.0001

            m = (beta1 * self.m) - ((1 - beta1) * filter_gradient)
            s = (beta2 * self.s) + ((1 - beta2) * (filter_gradient ** 2))
            m_hat = m / (1 - beta1)
            s_hat = s / (1 - beta2)
            self.m = m
            self.s = s

            m_bias = (beta1 * self.m_bias) - ((1 - beta1) * np.sum(output_gradient, axis = 0))
            s_bias = (beta2 * self.s_bias) + ((1 - beta2) * (np.sum(output_gradient, axis = 0) ** 2))
            m_bias_hat = m_bias / (1 - beta1)
            s_bias_hat = s_bias / (1 - beta2)
            self.m_bias = m_bias
            self.s_bias = s_bias

            self.kernels -= self.learning_rate * (m_hat / np.sqrt(s_hat + epsilon))
            self.biases -= self.learning_rate * (m_bias_hat / np.sqrt(s_bias_hat + epsilon))
        
        return input_gradient
