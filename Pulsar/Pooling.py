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
        self.mode = mode
        self.spatial_extent = spatial_extent
        self.stride = stride
        self.batch_size = batch_size
        self.depth = depth
        self.input_height = input_height
        self.input_width = input_width


    def forwardPass(self, batch, mode = 'forward'):
        current_batch_size = len(batch)

        x_positions = int((self.input_width - self.spatial_extent + self.stride) / self.stride)
        y_positions = int((self.input_height - self.spatial_extent + self.stride) / self.stride)

        batch = batch.reshape(current_batch_size, self.depth, self.input_height, self.input_width)
        output = batch[:current_batch_size, :self.depth, :self.input_height, :self.input_width].reshape(current_batch_size, self.depth, y_positions, self.spatial_extent, x_positions, self.spatial_extent).max(axis = (3, 5))
        #Output appears to be working

        # Only updates the pooling indexes during backpropagation; to speed up forward passes
        if mode == 'back':
            self.max_index = np.empty((current_batch_size, self.depth, y_positions, x_positions, 3))

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


    def backpropagate(self, batch, batch_labels = None, next_layer_weights = None, next_layer_grad = None):
        current_batch_size = len(batch)
        self.forwardPass(batch, mode = 'back')

        batch = batch.reshape(self.batch_size, self.depth, self.input_height, self.input_width)
        x_positions = int((self.input_width - self.spatial_extent + self.stride) / self.stride)
        y_positions = int((self.input_height - self.spatial_extent + self.stride) / self.stride)

        if next_layer_grad is not None:
            # Finds the output gradient if the next layer is dense - as dense layers do not return their input gradient
            try:
                fp = self.forwardPass(batch).reshape(current_batch_size, int((self.input_width - self.spatial_extent + self.stride) / self.stride) * int((self.input_height - self.spatial_extent + self.stride) / self.stride))
                derivative_matrix = Activation().derivativeReLU(fp)
                next_layer_grad = np.multiply(np.transpose(np.matmul(np.transpose(next_layer_weights), np.transpose(next_layer_grad))), derivative_matrix)
            except Exception as e:
                pass

            pooling_gradient = np.empty((self.batch_size, self.depth, self.input_height, self.input_width))

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

                            current_position_x = (x * self.stride) + self.spatial_extent - self.stride
                            current_position_y = (y * self.stride) + self.spatial_extent - self.stride
                            pooling_gradient[n][d][int(self.max_index[n][d][y][x][1]) + current_position_y][int(self.max_index[n][d][y][x][2]) + current_position_x] = grad

            return pooling_gradient
        
        else:
            print('Error: Gradients not received by pooling layer')
