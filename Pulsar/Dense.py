"""
Author: Sam Armstrong
Date: 2021

Description: Class for creating and interacting with dense (fully connected) network layers.
"""

class Dense:
    
    def __init__(self, Nin, Nout, initialization = 'He', activation = 'relu') -> None:
        self.Nin = Nin
        self.Nout = Nout