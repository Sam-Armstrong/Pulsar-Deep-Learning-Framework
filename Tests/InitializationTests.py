from context import Initialization
import unittest
import numpy as np

class TestInitializations(unittest.TestCase):
    
    def testRandomInit(self):
        i = Initialization()
        W = i.Random_init(10, 10)
        self.assertAlmostEqual(np.sum(W), 1.0, 3) # Checks that they sum to one


if __name__ == '__main__':
    unittest.main()