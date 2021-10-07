from keras.datasets import mnist # Import the MNIST dataset
from context import Pulsar
import time
import numpy as np

start_time = time.time()

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(60000, 784)
test_X = test_X.reshape(10000, 784)

# Converts training labels into one-hot format
rows = np.arange(train_y.size)
shape = (train_y.size, train_y.max() + 1)
one_hot = np.zeros(shape)
one_hot[rows, train_y] = 1
train_y = one_hot

p = Pulsar(epochs = 3, learning_rate = 0.0001, initialization = 'Xavier', penalty = 0.00003, loss = 'mean-squared', batch_size = 200)
p.convolution(28, 28, depth = 1, padding = 1, stride = 1, kernel_size = 3)
p.dense(784, 10, activation = 'relu')

p.train(train_X, train_y)

print("Training finished in %s seconds" % round((time.time() - start_time), 1))

predictions = p.batchPredict(test_X)
predictions = np.argmax(predictions, axis = 1)

correct = 0
total_samples = len(predictions)

for i in range(len(predictions)):
    pred = predictions[i]
    label = test_y[i]

    if pred == label:
        correct += 1

print('Percent correct: ', (correct / total_samples) * 100, '%')

print("Finished in %s seconds" % round((time.time() - start_time), 1))