from context import Pooling
import unittest
import numpy as np

class TestDenseLayers(unittest.TestCase):
    
    def testForwardPass(self):
        x = np.array([[-10, 20, 30, 1], [3, 4, 2, 1], [-10, 20, 30, 1], [3, 4, 2, 1]], np.int32)
        pooling = Pooling(4, 4, batch_size = 1)
        y = pooling.forwardPass(x)
        self.assertTrue(np.shape(y) == (1, 1, 2, 2)) # Checks that the ReLU activation function has been applied correctly


if __name__ == '__main__':
    unittest.main()