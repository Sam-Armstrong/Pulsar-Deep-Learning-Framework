from context import Pulsar
import time
import numpy as np

start_time = time.time()

from keras.datasets import mnist # Import the MNIST dataset

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(60000, 784)
test_X = test_X.reshape(10000, 784)

# Converts training labels into one-hot format
rows = np.arange(train_y.size)
shape = (train_y.size, train_y.max() + 1)
one_hot = np.zeros(shape)
one_hot[rows, train_y] = 1
train_y = one_hot

p = Pulsar(epochs = 1, learning_rate = 0.01, initialization = 'Xavier', penalty = 0.003, loss = 'cross-entropy')
p.dense(784, 100, activation = 'relu')
p.dense(100, 10, activation = 'softmax')
p.train(train_X, train_y)

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