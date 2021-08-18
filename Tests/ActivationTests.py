import unittest
import numpy as np
from context import Activation

class TestActivationFunctions(unittest.TestCase):

    global a
    a = Activation()

    # Tests that the ReLU activation function does not leave any negative numbers
    def testReLU(self):
        x = np.array([[-1, 2, 3], [4, -5, 6]], np.int32)
        y = a.ReLU(x)
        self.assertFalse(np.any(y < 0))

    # Tests that the ReLU derivative only returns matrices of ones and zeros
    def testDerivationReLU(self):
        x = np.array([[-1, 2, 3], [4, -5, 6]], np.int32)
        y = a.derivationReLU(x)
        self.assertFalse(np.all(y != 0))
        self.assertFalse(np.all(y != 1))

    # Tests that all the values produced by the sigmoid activation function are within the range 0 to 1
    def testSigmoid(self):
        x = np.array([[-1, 2, 3], [4, -5, 6]], np.int32)
        y = a.logisticSigmoid(x)
        self.assertTrue(np.all(y < 1))
        self.assertTrue(np.all(y > 0))

    # Tests that the derivative of sigmoid never returns negative values (as the gradient of sigmoid is always >= 0)
    def testDerivationSigmoid(self):
        x = np.array([[-1, 2, 3], [4, -5, 1]], np.int32)
        y = a.derivationSigmoid(x)
        self.assertTrue(np.all(y >= 0))

    # Tests that the output of softmax is a probability distribution
    def testSoftmax(self):
        x = np.array([1, 5, 2, 11, 10], np.int32)
        y = a.softmax(x)
        self.assertTrue(np.all(y < 1))
        self.assertTrue(np.all(y > 0))
        self.assertAlmostEqual(np.sum(y), 1.0, 3)


if __name__ == '__main__':
    unittest.main()