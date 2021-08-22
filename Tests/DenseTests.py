from context import Dense
import unittest
import numpy as np

class TestDenseLayers(unittest.TestCase):
    
    def testForwardPass(self):
        x = np.array([[-10, 20, 30], [3, 4, 2]], np.int32)
        dense = Dense(3, 2, initialization = 'random')
        y = dense.forwardPass(x)
        self.assertTrue(y.shape == (2, 2))
        self.assertTrue(np.all(y) >= 0) # Checks that the ReLU activation function has been applied correctly


if __name__ == '__main__':
    unittest.main()