"""
Author: Sam Armstrong
Date: 2021

Description: Class providing methods for initializing the weights in a neural network.
"""

import numpy as np

class Initialization:

    def __init__(self) -> None:
        pass

    # Defines a random weight initialization
    def Random_init(self, Nin, Nout):
        W = np.random.uniform(0, 1, (Nout, Nin))
        return W / np.sum(W) # Normalizes the weights so they sum to 1

    # Defines a Xavier weight initialization
    def Xavier_init(self, Nin, Nout):
        W = np.random.randn(Nout, Nin) * np.sqrt(1 / (Nin + Nout))
        return W / np.sum(W)

    # Defines a He initialization
    def He_init(self, Nin, Nout):
        W = np.random.randn(Nout, Nin) * np.sqrt(2 / (Nin + Nout))
        return W