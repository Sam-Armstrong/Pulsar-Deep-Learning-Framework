"""
Author: Sam Armstrong
Date: 2021

Description: Class for creating and interacting with pooling layers.
"""

import numpy as np
from Activation import Activation

class Pooling:
    
    def __init__(self, input_height, input_width, mode = 'max', spatial_extent = 2, stride = 2, 
                 batch_size = 200, depth = 1):
        # Defines the attributes of this pooling object
        self.mode = mode
        self.spatial_extent = spatial_extent
        self.stride = stride
        self.batch_size = batch_size
        self.depth = depth
        self.input_height = input_height
        self.input_width = input_width
        self.a = Activation()


    # Defines a forward pass through a pooling layer to reduce the size of a given feature map
    def forward(self, batch, mode ='forward'):
        if np.shape(batch) == 4:
            current_batch_size = len(batch)
        else:
            current_batch_size = self.batch_size

        # Finds the size of the layer output
        x_positions = int((self.input_width - self.spatial_extent + self.stride) / self.stride)
        y_positions = int((self.input_height - self.spatial_extent + self.stride) / self.stride)

        # Ensures the incoming data is in the correct shape
        batch = batch.reshape(current_batch_size, self.depth, self.input_height, self.input_width)
        output = batch[:current_batch_size, :self.depth, :self.input_height, :self.input_width].reshape(current_batch_size, self.depth, y_positions, self.spatial_extent, x_positions, self.spatial_extent).max(axis = (3, 5))

        # Only updates the pooling indexes during backpropagation; to speed up forward passes
        if mode == 'back':
            self.max_index = np.empty((current_batch_size, self.depth, y_positions, x_positions, 3))

            # Loops through every datapoint in order to log the positions of max values that become the output
            for n in range(current_batch_size):
                data = batch[n]
                for d in range(self.depth):
                    for y in range(y_positions):
                        for x in range(x_positions):
                            current_y = y * self.spatial_extent
                            current_x = x * self.spatial_extent
                            current_data = data[:, current_y:current_y+self.spatial_extent, current_x:current_x+self.spatial_extent]
                            max_value = np.amax(current_data)

                            indexes = np.where(current_data == max_value)

                            # Stores the index of each of the max values
                            self.max_index[n][d][y][x][0] = indexes[0][0]
                            self.max_index[n][d][y][x][1] = indexes[1][0]
                            self.max_index[n][d][y][x][2] = indexes[2][0]
        
        return output


    # Backpropagates the error through the pooling layer
    def backward(self, batch, batch_labels = None, next_layer_weights = None, next_layer_grad = None):
        current_batch_size = len(batch)
        self.forward(batch, mode = 'back')

        # Unflattens the batch
        batch = batch.reshape(self.batch_size, self.depth, self.input_height, self.input_width)

        # Finds the size of the output of this layer
        x_positions = int((self.input_width - self.spatial_extent + self.stride) / self.stride)
        y_positions = int((self.input_height - self.spatial_extent + self.stride) / self.stride)

        if next_layer_grad is not None:
            # Finds the output gradient if the next layer is dense - as dense layers do not return their input gradient
            try:
                fp = self.forward(batch).reshape(current_batch_size, self.depth * int((self.input_width - self.spatial_extent + self.stride) / self.stride) * int((self.input_height - self.spatial_extent + self.stride) / self.stride))
                derivative_matrix = self.a.derivativeReLU(fp)
                next_layer_grad = np.multiply(np.transpose(np.matmul(np.transpose(next_layer_weights), np.transpose(next_layer_grad))), derivative_matrix)
            except Exception as e:
                print(e)
                pass

            pooling_gradient = np.empty((self.batch_size, self.depth, self.input_height, self.input_width))

            # Loops through every datapoint in the batch to find the location of the 'max' values that were used in the forward pass
            # The gradient is then pass backward in these positions to train precceding layers
            for n in range(len(batch)):
                for d in range(self.depth):
                    for y in range(y_positions):
                        for x in range(x_positions):
                            try:
                                # When the next layer is a Convolution
                                grad = next_layer_grad[n][d][y][x]
                            except:
                                # When the next layer is Dense
                                grad = next_layer_grad[n][(d * y_positions * x_positions) + (y * x_positions) + x - 1]

                            # Finds the current position on the output feature map (the reduced feature map)
                            current_position_x = (x * self.stride) + self.spatial_extent - self.stride
                            current_position_y = (y * self.stride) + self.spatial_extent - self.stride

                            # Assigns the gradient to the correct position in the original feature map
                            pooling_gradient[n][d][int(self.max_index[n][d][y][x][1]) + current_position_y][int(self.max_index[n][d][y][x][2]) + current_position_x] = grad

            return pooling_gradient
        
        else:
            # If no output gradients are passed
            print('Error: Gradients not received by pooling layer')
