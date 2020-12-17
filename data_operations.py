import matplotlib.pyplot as plt
import numpy as np
# import keras
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
index = np.where(Y_train == 1)
X_train = X_train[index]


class BatchedData():
    # This class was prepared to create easy access to data.
    # It also take care of batching and standardising provided data

    def __init__(self, batch_size, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test):
        self.X_train = X_train.reshape([X_train.shape[0], 28, 28])
        self.Y_train = Y_train
        self.X_test = X_test.reshape([X_test.shape[0], 784])
        self.Y_test = Y_test

        self.batch_size = batch_size
        self.index = np.random.randint(0, self.X_train.shape[0] - self.batch_size)

        self.x_batch = self.X_train[self.index:self.index + self.batch_size, :].reshape([self.batch_size, 784])
        self.y_batch = self.Y_train[self.index:self.index + self.batch_size]
        self.noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, 100])
        self.x_true = self.x_batch

    def next_batch(self):
        # Method to generate new batch from data
        old_index = self.index

        while True:

            if old_index == self.index:
                self.index = np.random.randint(0, self.X_train.shape[0] - self.batch_size)

                self.x_batch = self.X_train[self.index:self.index + self.batch_size, :].reshape([self.batch_size, 784])
                self.y_batch = self.Y_train[self.index:self.index + self.batch_size]
                self.x_true = self.x_batch
            else:
                break

            return self.x_batch

    def show(self):
        # Print example of an data
        i = np.random.randint(0, self.batch_size)
        plt.imshow(self.x_true[i].reshape([28, 28]), cmap="Greys")
