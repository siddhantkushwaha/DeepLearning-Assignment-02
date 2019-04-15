import os

from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# %%

def get_images(path):
    data = []
    for root, _, files in os.walk(path):
        print('Reading directory -> ' + root)
        for file in files:
            if '.jpg' in file:
                img = Image.open(root + '/' + file)
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.asarray(img)
                data.append(img)
    return np.array(data)


# %%

class FaceGAN:
    def __init__(self, image_width, image_height, channels):
        self.image_width = image_width
        self.image_height = image_height

        self.channels = channels

        self.image_shape = (self.image_width, self.image_height, self.channels)

        self.random_noise_dimension = 100

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        self.generator = self.build_generator()

        random_input = Input(shape=(self.random_noise_dimension,))

        generated_image = self.generator(random_input)

        self.discriminator.trainable = False

        validity = self.discriminator(generated_image)

        self.combined = Model(random_input, validity)
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256 * 8 * 8, activation="relu", input_dim=self.random_noise_dimension))
        model.add(Reshape((8, 8, 256)))

        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        input_noise = Input(shape=(self.random_noise_dimension,))

        generated_image = model(input_noise)

        return Model(input_noise, generated_image)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.image_shape, padding="same"))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))

        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        input_image = Input(shape=self.image_shape)

        validity = model(input_image)

        return Model(input_image, validity)

    def train(self, path, epochs, batch_size, save_images_interval):

        training_data = get_images(path)

        training_data = training_data / 127.5 - 1.

        labels_for_real_images = np.ones((batch_size, 1))
        labels_for_generated_images = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            indices = np.random.randint(0, training_data.shape[0], batch_size)
            real_images = training_data[indices]

            random_noise = np.random.normal(0, 1, (batch_size, self.random_noise_dimension))
            generated_images = self.generator.predict(random_noise)

            discriminator_loss_real = self.discriminator.train_on_batch(real_images, labels_for_real_images)

            discriminator_loss_generated = self.discriminator.train_on_batch(generated_images,
                                                                             labels_for_generated_images)

            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_generated)

            generator_loss = self.combined.train_on_batch(random_noise, labels_for_real_images)
            print("%d [Discriminator -> loss: %f, acc.: %.2f%%] [Generator -> loss: %f]" % (
                epoch, discriminator_loss[0], 100 * discriminator_loss[1], generator_loss))

            if epoch % save_images_interval == 0:
                self.plot_images(epoch + 1)

    def plot_images(self, epoch):
        rows, columns = 5, 5
        noise = np.random.normal(0, 1, (rows * columns, self.random_noise_dimension))
        generated_images = self.generator.predict(noise)

        generated_images = 0.5 * generated_images + 0.5

        figure, axis = plt.subplots(rows, columns)
        image_count = 0
        for row in range(rows):
            for column in range(columns):
                axis[row, column].imshow(generated_images[image_count, :], cmap='spring')
                axis[row, column].axis('off')
                image_count += 1

        plt.savefig('epoch-%d' % epoch)


# %%

face_gan = FaceGAN(128, 128, 3)
face_gan.train(path='faces94', epochs=4000, batch_size=32, save_images_interval=100)
