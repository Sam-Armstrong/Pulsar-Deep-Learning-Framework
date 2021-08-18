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


if __name__ == '__main__':
    unittest.main()