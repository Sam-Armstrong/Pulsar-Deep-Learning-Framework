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

    def forwardPass(self, batch):
        current_batch_size = len(batch)

        batch = batch.reshape(current_batch_size, self.depth, self.input_height, self.input_width)
        x_positions = int((self.input_width - self.spatial_extent + self.stride) / self.stride)
        y_positions = int((self.input_height - self.spatial_extent + self.stride) / self.stride)

        output = np.zeros((current_batch_size, self.depth, y_positions, x_positions))
        self.max_index = np.empty((current_batch_size, self.depth, y_positions, x_positions, 3))

        for n in range(current_batch_size):
            data = batch[n]
            for d in range(self.depth):
                for y in range(y_positions):
                    for x in range(x_positions):
                        current_data = data[:, y:y+self.spatial_extent, x:x+self.spatial_extent]
                        max_value = np.amax(current_data)
                        output[n][d][y][x] = max_value

                        indexes = np.where(current_data == max_value)
                        self.max_index[n][d][y][x][0] = indexes[0][0] + d
                        self.max_index[n][d][y][x][1] = indexes[1][0] + y
                        self.max_index[n][d][y][x][2] = indexes[2][0] + x

        return output


    def backpropagate(self, batch, batch_labels = None, next_layer_weights = None, next_layer_grad = None):
        
        current_batch_size = len(batch)

        self.forwardPass(batch)

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
                data = batch[n]
                for d in range(self.depth):
                    for y in range(y_positions):
                        for x in range(x_positions):
                            try:
                                # Works when the next layer is a Convolution
                                grad = next_layer_grad[n][d][y][x]
                            except:
                                # Works when the next layer is Dense
                                grad = next_layer_grad[n][(d * y_positions * x_positions) + (y * x_positions) + x]

                            pooling_gradient[n][int(self.max_index[n][d][y][x][0])][int(self.max_index[n][d][y][x][1])][int(self.max_index[n][d][y][x][2])] = grad

            return pooling_gradient
        
        else:
            print('Error: Gradients not received by pooling layer')
