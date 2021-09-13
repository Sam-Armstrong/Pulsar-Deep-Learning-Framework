"""
Author: Sam Armstrong
Date: 2021

Description: Class for creating and interacting with convolutional network layers.
"""

from Initialization import Initialization
from Activation import Activation
import numpy as np
import math
from scipy import signal

class Convolution:
    
    def __init__(self, input_height, input_width, kernel_size = 3, depth = 1, input_depth = 1, batch_size = 200,
                 stride = 1, padding = 0, activation = 'relu', learning_rate = 0.001) -> None:
        
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.activation = activation
        self.stride = stride
        self.padding = padding
        # kernel_size: size of the matrix inside each kernel (filter size)
        self.depth = depth # The number of kernels (depth of the output)
        self.input_depth = input_depth # The depth of the input (rbg? (3))
        self.input_height = input_height
        self.input_width = input_width
        self.output_shape = (batch_size, depth, (input_height + (padding * 2)) - kernel_size + 1, (input_width + (padding * 2)) - kernel_size + 1) # The shape of the output from the layer
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size) # The shape of the kernels (filters)
        
        # Initialize the filters and biases
        self.kernels = np.random.randn(*self.kernels_shape) * (1 / (kernel_size * kernel_size * input_depth)) # * Collects argument(s) into a tuple
        self.biases = np.zeros((depth, int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride), int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride))) #(input_height - kernel_size + 1, input_width - kernel_size + 1) #(*self.output_shape)


    def forwardPass(self, batch):
        current_batch_size = len(batch)
        batch = batch.reshape(current_batch_size, self.input_depth, self.input_height, self.input_width)
        output_batch = np.empty((current_batch_size, self.depth, int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride), int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride)))

        # Loop through the number of output kernels
        for n in range(current_batch_size):
            current_output_batch = np.empty((self.depth, int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride), int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride)))
            data = batch[n]

            # Applies the chosen padding
            if self.padding != 0:
                data = np.pad(data, self.padding)
                for i in range(self.padding):
                    data = np.delete(data, 0, axis = 0)
                    data = np.delete(data, len(data) - 1, axis = 0)

            for i in range(self.depth):
                output = np.copy(self.biases[i]) # The ouput is the bias + the convolution output
                

                for d in range(self.input_depth):
                    filter = self.kernels[i][d]
                    output += signal.correlate(data[d], filter, mode = 'valid') #filter[i]

                current_output_batch[i] = output

            output_batch[n] = current_output_batch

            # Applies the activation function
            output_batch = Activation(activation = self.activation).activate(output_batch)
        
        return output_batch


    def backpropagate(self, batch, batch_labels = None, next_layer_weights = None, next_layer_grad = None):

        current_batch_size = len(batch)

        if len(next_layer_grad.shape) == 4:
            next_layer_grad = next_layer_grad.reshape(current_batch_size, next_layer_grad.shape[3] * next_layer_grad.shape[2] * next_layer_grad.shape[1])

        # Finds the output gradient if the next layer is dense - as dense layers do not return their input gradient
        try:
            fp = self.forwardPass(batch).reshape(current_batch_size, int((self.input_width - self.kernel_size + self.stride) / self.stride) * int((self.input_height - self.kernel_size + self.stride) / self.stride))
            derivative_matrix = Activation().derivativeReLU(fp)
            output_gradient = np.multiply(np.transpose(np.matmul(np.transpose(next_layer_weights), np.transpose(next_layer_grad))), derivative_matrix)
        except Exception as e:
            output_gradient = next_layer_grad

        # dL/dF = Convolution of the input and the loss gradient
        # dL/dX = Full Convolution of the 180 degree rotated filter and the loss gradient
        
        if len(output_gradient.shape) == 2:
            output_gradient_size = self.input_width + (self.padding * 2) - self.kernel_size + 1
            output_gradient = output_gradient.reshape(self.batch_size, self.depth, (self.input_height + (self.padding * 2)) - self.kernel_size + 1, (self.input_width + (self.padding * 2)) - self.kernel_size + 1)###
        else:
            output_gradient_size = output_gradient.shape[len(output_gradient - 1)]

        batch = batch.reshape(self.batch_size, self.input_depth, self.input_height, self.input_width)
        filter_gradient = np.empty((self.depth, self.input_depth, self.kernel_size, self.kernel_size))
        input_gradient = np.empty((self.batch_size, self.input_depth, self.input_height, self.input_width))

        max_padding = output_gradient_size - 1 # The largest padding that can be applied

        for n in range(self.batch_size):
            current_filter_batch = np.empty((self.depth, self.input_depth, int((self.input_height - output_gradient_size + self.stride) / self.stride), int((self.input_width - output_gradient_size + self.stride) / self.stride)))
            input_batch_grad = np.empty((self.input_depth, int(((self.kernel_size + (max_padding * 2)) - output_gradient_size + self.stride) / self.stride), int(((self.kernel_size + (max_padding * 2)) - output_gradient_size + self.stride) / self.stride)))
            data = batch[n]

            for i in range(self.depth):  
                filter_x_positions = int((self.input_width - output_gradient_size + self.stride) / self.stride)
                filter_y_positions = int((self.input_height - output_gradient_size + self.stride) / self.stride)
                filter_output = np.zeros((self.input_depth, filter_y_positions, filter_x_positions))

                input_x_positions = int(((self.kernel_size + (max_padding * 2)) - output_gradient_size + self.stride) / self.stride)
                input_y_positions = int(((self.kernel_size + (max_padding * 2)) - output_gradient_size + self.stride) / self.stride)
                input_grad = np.zeros((input_y_positions, input_x_positions))
                
                for d in range(self.input_depth):
                    filter = self.kernels[i][d]
                    filter_output[d] = signal.correlate(data[d], output_gradient[n][i], mode = 'valid').reshape(filter_y_positions, filter_x_positions)
                    input_batch_grad[d] += signal.convolve(output_gradient[n][i], filter, mode = 'full')

                current_filter_batch[i] = filter_output # (1, 3, 3)

            filter_gradient = current_filter_batch # (3, 1, 3, 3)
            input_gradient[n] = input_batch_grad # / depth?

        self.kernels -= (self.learning_rate * filter_gradient) / current_batch_size
        self.biases -= (self.learning_rate * np.sum(output_gradient, axis = 0)) / current_batch_size
        
        return input_gradient
