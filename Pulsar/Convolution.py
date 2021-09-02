"""
Author: Sam Armstrong
Date: 2021

Description: Class for creating and interacting with dense (fully connected) network layers.
"""

from numpy.lib.function_base import gradient
from Initialization import Initialization
from Activation import Activation
from Loss import Loss
import numpy as np
import time
import math

from keras.datasets import mnist # Import the MNIST dataset

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
        self.kernels = np.random.randn(*self.kernels_shape) # * Collects argument(s) into a tuple
        self.biases = np.random.randn(depth, int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride), int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride)) #(input_height - kernel_size + 1, input_width - kernel_size + 1) #(*self.output_shape)


    def forwardPass(self, batch):
        current_batch_size = len(batch)
        batch = batch.reshape(current_batch_size, self.input_depth, self.input_height, self.input_width)
        output_batch = np.empty((current_batch_size, self.depth, int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride), int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride)))

        # Loop through the number of output kernels
        for n in range(current_batch_size):
            current_output_batch = np.empty((self.depth, int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride), int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride)))
            data = batch[n]

            if self.padding != 0:
                data = np.pad(data, self.padding)
                data = np.delete(data, 0, axis = 0)
                data = np.delete(data, len(data) - 1, axis = 0)

            for i in range(self.depth):
                output = np.copy(self.biases[i]) # The ouput is the bias + the convolution output
            
                # Calculate the cross-correlation of the input and the filter
                x_positions = int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride)
                y_positions = int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride)
                filter = self.kernels[i]

                # Loop through all possible positions for the filter on the image
                for y in range(y_positions):
                    for x in range(x_positions):
                        if x % self.stride == 0 and y % self.stride == 0:
                            current_data = data[:, y:y+self.kernel_size, x:x+self.kernel_size]
                            current_output = np.sum(filter * current_data)
                            output[x][y] += current_output

                current_output_batch[i] = output

            output_batch[n] = current_output_batch

            # Applies the activation function
            output_batch = Activation(activation = self.activation).activate(output_batch)
        
        return output_batch


    def backpropagate(self, batch, batch_labels = None, next_layer_weights = None, next_layer_grad = None):

        # Finds the output gradient if the next layer is dense - as dense layers do not return their input gradient
        try:
            fp = self.forwardPass(batch).reshape(200, 676)
            derivative_matrix = Activation().derivativeReLU(fp)
            next_layer_grad = np.multiply(np.transpose(np.matmul(np.transpose(next_layer_weights), np.transpose(next_layer_grad))), derivative_matrix)
        except Exception as e:
            print(e)

        # dL/dF = Convolution of the input and the loss gradient

        next_layer_grad = np.sum(next_layer_grad, axis = 0)
        next_layer_grad = next_layer_grad / self.batch_size

        if len(next_layer_grad.shape) == 1:
            output_gradient_size = int(math.sqrt(len(next_layer_grad)))
            next_layer_grad = next_layer_grad.reshape(1, output_gradient_size, output_gradient_size)###
        else:
            output_gradient_size = next_layer_grad.shape[len(next_layer_grad - 1)]

        batch = batch.reshape(self.batch_size, self.input_depth, self.input_height, self.input_width)
        filter_gradient = np.empty((self.depth, self.kernel_size, self.kernel_size))
        input_gradient = np.empty((self.batch_size, self.input_depth, self.input_height, self.input_width))


        for n in range(self.batch_size):
            current_output_batch = np.empty((self.depth, int((self.input_height - output_gradient_size + self.stride) / self.stride), int((self.input_width - output_gradient_size + self.stride) / self.stride)))
            data = batch[n]

            for i in range(self.depth):            
                # Calculate the cross-correlation of the input and the filter
                x_positions = int((self.input_width - output_gradient_size + self.stride) / self.stride)
                y_positions = int((self.input_height - output_gradient_size + self.stride) / self.stride)
                output = np.empty((y_positions, x_positions))

                # Loop through all possible positions for the filter on the image
                for y in range(y_positions):
                    for x in range(x_positions):
                        if x % self.stride == 0 and y % self.stride == 0:
                            current_data = data[:, y:y+output_gradient_size, x:x+output_gradient_size]
                            current_output = np.sum(next_layer_grad * current_data)
                            output[x][y] += current_output

                current_output_batch[i] = output

            filter_gradient += current_output_batch


        # dL/dX = Full Convolution of the 180 degree rotated filter and the loss gradient
        
        max_padding = output_gradient_size - 1 # The largest padding that can be applied

        for n in range(self.batch_size):
            current_output_batch = np.empty((self.depth, int(((self.kernel_size + (max_padding * 2)) - output_gradient_size + self.stride) / self.stride), int(((self.kernel_size + (max_padding * 2)) - output_gradient_size + self.stride) / self.stride)))

            for i in range(self.depth):            
                # Calculate the cross-correlation of the input and the filter
                x_positions = int(((self.kernel_size + (max_padding * 2)) - output_gradient_size + self.stride) / self.stride)
                y_positions = int(((self.kernel_size + (max_padding * 2)) - output_gradient_size + self.stride) / self.stride)
                output = np.empty((y_positions, x_positions))

                filter = self.kernels[i]
                rotated_filter = np.flip(filter, axis = 0)
                rotated_filter = np.pad(rotated_filter, max_padding)
                while len(rotated_filter) > self.kernel_size:
                    rotated_filter = np.delete(rotated_filter, 0, axis = 0)
                    rotated_filter = np.delete(rotated_filter, len(rotated_filter) - 1, axis = 0)

                # Loop through all possible positions for the filter on the image
                for y in range(y_positions):
                    for x in range(x_positions):
                        if x % self.stride == 0 and y % self.stride == 0:
                            current_data = rotated_filter[:, y:y+output_gradient_size, x:x+output_gradient_size]
                            current_output = np.sum(next_layer_grad * current_data)
                            output[x][y] += current_output

                current_output_batch[i] = output

            input_gradient[n] = current_output_batch

        self.kernels -= self.learning_rate * filter_gradient
        self.biases -= self.learning_rate * next_layer_grad
        return input_gradient

