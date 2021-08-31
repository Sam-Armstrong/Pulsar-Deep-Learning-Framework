"""
Author: Sam Armstrong
Date: 2021

Description: Class for creating and interacting with dense (fully connected) network layers.
"""

from Initialization import Initialization
from Activation import Activation
from Loss import Loss
import numpy as np

from keras.datasets import mnist # Import the MNIST dataset

class Convolution:
    
    def __init__(self, input_height, input_width, kernel_size = 3, depth = 1, input_depth = 1, batch_size = 200, stride = 1) -> None:
        self.kernel_size = kernel_size
        self.batch_size = batch_size

        self.stride = stride
        # kernel_size: size of the matrix inside each kernel (filter size)
        self.depth = depth # The number of kernels (depth of the output)
        self.input_depth = input_depth # The depth of the input (rbg? (3))
        self.input_height = input_height
        self.input_width = input_width
        self.output_shape = (batch_size, depth, input_height - kernel_size + 1, input_width - kernel_size + 1) # The shape of the output from the layer
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size) # The shape of the kernels (filters)
        
        # Initialize the filters and biases
        self.kernels = np.random.randn(*self.kernels_shape) # * Collects argument(s) into a tuple
        self.biases = np.random.randn(input_height - kernel_size + 1, input_width - kernel_size + 1) #(*self.output_shape)


    def forwardPass(self, batch):
        batch = batch.reshape(self.batch_size, self.input_depth, self.input_height, self.input_width)
        output_batch = np.empty((self.batch_size, self.depth, int((self.input_height - self.kernel_size + self.stride) / self.stride), int((self.input_width - self.kernel_size + self.stride) / self.stride)))

        # Loop through the number of output kernels
        for n in range(self.batch_size):
            current_output_batch = np.empty((self.depth, int((self.input_height - self.kernel_size + self.stride) / self.stride), int((self.input_width - self.kernel_size + self.stride) / self.stride)))

            for i in range(self.depth):
                input = batch[n]
                output = np.copy(self.biases) # The ouput is the bias + the convolution output
            
                # Calculate the cross-correlation of the input and the filter
                x_positions = (self.input_width - self.kernel_size + 1)
                y_positions = (self.input_height - self.kernel_size + 1)
                filter = self.kernels[i]

                # Loop through all possible positions for the filter on the image
                for y in range(y_positions):
                    for x in range(x_positions):
                        current_data = input[:, y:y+self.kernel_size, x:x+self.kernel_size]
                        current_output = np.sum(filter * current_data)
                        output[x][y] += current_output

                current_output_batch[i] = output

            output_batch[n] = current_output_batch

            # Activate
            output_batch = Activation().activate(output_batch)
        
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
c = Convolution(28, 28, batch_size = 200, depth = 2)
y = c.forwardPass(train_X[0:200])
#print(y)
print(np.shape(y))
