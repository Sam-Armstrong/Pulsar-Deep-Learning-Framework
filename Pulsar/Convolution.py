"""
Author: Sam Armstrong
Date: 2021

Description: Class for creating and interacting with dense (fully connected) network layers.
"""

from Initialization import Initialization
from Activation import Activation
from Loss import Loss
import numpy as np
import time

from keras.datasets import mnist # Import the MNIST dataset

class Convolution:
    
    def __init__(self, input_height, input_width, kernel_size = 3, depth = 1, input_depth = 1, batch_size = 200,
                 stride = 1, padding = 0, activation = 'relu') -> None:
        
        self.kernel_size = kernel_size
        self.batch_size = batch_size

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
        batch = batch.reshape(self.batch_size, self.input_depth, self.input_height, self.input_width)
        output_batch = np.empty((self.batch_size, self.depth, int(((self.input_height + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride), int(((self.input_width + (self.padding * 2)) - self.kernel_size + self.stride) / self.stride)))

        # Loop through the number of output kernels
        for n in range(self.batch_size):
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


    """def backpropagate(self, batch, batch_labels = None, next_layer_weights = None, next_layer_grad = None):
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
        self.biases += (sum(delta) * self.lr) / len(batch)"""


(train_X, train_y), (test_X, test_y) = mnist.load_data()
c = Convolution(28, 28, batch_size = 10, depth = 2, stride = 1, padding = 1)
start_time = time.time()
y = c.forwardPass(train_X[0:10])
#print(y)
print(np.shape(y))
print("Finished in %s seconds" % round((time.time() - start_time), 1))
