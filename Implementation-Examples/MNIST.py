from keras.datasets import mnist # Import the MNIST dataset
from context import Pulsar
import time
import numpy as np

start_time = time.time()

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(60000, 1, 784)

# Converts training labels into one-hot format
rows = np.arange(train_y.size)
shape = (train_y.size, train_y.max() + 1)
one_hot = np.zeros(shape)
one_hot[rows, train_y] = 1
train_y = one_hot

#print(train_y)

p = Pulsar()
p.dense(784, 400)
p.dense(400, 100)
p.dense(100, 10)
p.train(train_X, train_y)



print("Finished in %s seconds" % round((time.time() - start_time), 1))