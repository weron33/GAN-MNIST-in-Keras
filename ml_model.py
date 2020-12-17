import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense, Reshape, Conv2D, Flatten
from data_operations import BatchedData


class Generator():
    # This class contains model for generator.
    # It process 100 shape input into 28x28 output.

    def __init__(self):
        self.dropout = 0.4
        self.dim = 28
        self.depth = 4

        self.model = Sequential()
        n_nodes = 28 * 28
        self.model.add(Dense(n_nodes * 3, activation='relu', input_dim=100))

        self.model.add(Reshape((3, 28, 28)))
        self.model.add(Conv2D(3, (3, 3), strides=(2, 2), padding='same'))
        self.model.add(Flatten())

        self.model.add(Dense(n_nodes, activation='sigmoid'))

    def summary(self):
        print(self.model.summary())


class Discriminator():
    # This class contains model for discriminator.
    # It process 784 shape input into 1 output.

    def __init__(self):
        self.depth = 2
        self.dropout = 0.4
        self.input_dim = (28, 28, 1)

        self.model = Sequential()
        self.model.add(Dense(784 * 3, activation='relu', input_dim=784))

        self.model.add(Reshape((3, 28, 28)))
        self.model.add(Conv2D(3, (3, 3), strides=(2, 2), padding='same'))
        self.model.add(Flatten())

        self.model.add(Dense(1, activation='sigmoid'))

    def summary(self):
        print(self.model.summary())

    def compile(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


class AdversarialModel():
    # This class is like connector for Generator class and Discriminator class.
    # As an input it gets number of epochs to train.
    # It can display summary of model, compile model, generate fake images, train and predict if image is fake or real.
    # Additional it was implemented to save and load model, as well as plot its train history.

    def __init__(self, epochs=5000):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.discriminator.model.trainable = False
        self.batched_data = BatchedData(batch_size=16)
        self.epochs = epochs
        self.adv_history = []
        self.dis_history = []

    def summary(self):
        print('---------------------------- GENERATOR ----------------------------')
        print(self.generator.summary())
        print(' ')
        print('-------------------------- DISCRIMINATOR --------------------------')
        print(self.discriminator.summary())

    def compile(self):
        self.adv_model = Sequential()

        self.adv_model.add(self.generator.model)
        self.adv_model.add(self.discriminator.model)

        # Setting discriminator again so it will train among side generator but it's loss won't affect generator's loss
        self.discriminator = Discriminator()
        self.adv_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def generate_images(self, noise):
        images_fake = self.generator.model.predict(noise)

        # Connecting fake and real images in batched_data
        self.batched_data.x_batch = np.concatenate((self.batched_data.x_batch, images_fake))
        self.batched_data.y_batch = np.ones([2 * self.batched_data.batch_size, 1])
        self.batched_data.y_batch[self.batched_data.batch_size:, :] = 0

        return images_fake

    def train(self, epochs=500):
        self.compile()
        self.batched_data.next_batch()

        self.discriminator.compile()

        # Starting training
        for epoch in range(epochs):
            noise_gen = np.random.uniform(-1.0, 1.0, size=[self.batched_data.batch_size, 100])

            self.batched_data.next_batch()
            _ = self.generate_images(noise_gen)
            x = self.batched_data.x_batch

            y = np.ones([2 * self.batched_data.batch_size, 1])
            y[self.batched_data.batch_size:, :] = 0

            d_loss = self.discriminator.model.train_on_batch(x, y)

            noise = np.random.uniform(-1.0, 1.0, size=[self.batched_data.batch_size, 100])
            y = np.ones([self.batched_data.batch_size, 1])

            a_loss = self.adv_model.train_on_batch(noise, y)

            # Saving train history
            self.adv_history.append(a_loss[0])
            self.dis_history.append(d_loss[0])

    def predict(self, noise):
        images_fake = self.generate_images(noise)
        accuracy = self.discriminator.model.predict(images_fake)
        return accuracy

    def save(self, filename):
        self.generator.model.save(filename + '-gen')
        self.discriminator.model.save(filename + '-dis')
        self.adv_model.save(filename)

    def load(self, filename):
        self.generator.model = keras.models.load_model(filename + '-gen')
        self.discriminator.model = keras.models.load_model(filename + '-dis')
        self.adv_model = keras.models.load_model(filename)

    def plot(self):
        adv_history = np.array(self.adv_history)
        dis_history = np.array(self.dis_history)

        plt.plot(range(len(adv_history)), adv_history, range(len(dis_history)), dis_history)
        plt.legend(['a_loss', 'd_loss'])