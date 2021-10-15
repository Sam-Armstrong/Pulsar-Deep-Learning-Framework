from context import Convolution
import unittest
import numpy as np
from keras.datasets import mnist # Import the MNIST dataset

class TestConvolutionLayers(unittest.TestCase):

    global train_X
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    
    # Tests that the convolutional layer outputs a feature map that is slightly reduced in size compared to the input images
    # Also tests that the same number of samples are output as the batch size
    def testForwardPass1(self):
        c = Convolution(28, 28, batch_size = 200)
        y = c.forwardPass(train_X[0:200])
        print(np.shape(y))
        self.assertTrue(np.shape(y) == (200, 1, 26, 26))

    # Testing for a batch size of 1
    def testForwardPass2(self):
        c = Convolution(28, 28, batch_size = 1, depth = 1)
        y = c.forwardPass(train_X[0])
        self.assertTrue(np.shape(y) == (1, 1, 26, 26))


if __name__ == '__main__':
    unittest.main()