import os
import numpy as np
import pandas as pd
from PIL import Image

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dense
from keras.models import Model


# %%

def get_images(path):
    input_data = []
    labels = []
    for root, _, files in os.walk(path):
        print('Reading directory -> ' + root)
        for file in files:
            if '.jpg' in file:
                label = root + '/' + file

                img = Image.open(label)
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.asarray(img)

                input_data.append(img)
                labels.append(label)

    return np.array(input_data) / 255.0, np.reshape(labels, (len(labels), 1))


# %%

x, y, channels = 128, 128, 3
input_img = Input(shape=(x, y, channels))


# %%

class AutoEncoder:
    def __init__(self, input_img):
        encoder = self.build_encoder(input_img)
        decoder = self.build_decoder(encoder)

        self.encoder_model = Model(input_img, encoder)

        self.autoencoder_model = Model(input_img, decoder)

    def build_encoder(self, input_img):
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        flatten = Flatten()(pool3)
        dense1 = Dense(128, activation='relu')(flatten)
        encoder = Dense(100, activation='relu')(dense1)
        return encoder

    def build_decoder(self, encoder):
        dense3 = Dense(128, activation='relu')(encoder)
        dense4 = Dense(128 * (16 * 16), activation='relu')(dense3)
        reshape = Reshape((16, 16, 128))(dense4)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(reshape)
        up1 = UpSampling2D((2, 2))(conv4)
        conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
        up2 = UpSampling2D((2, 2))(conv5)
        conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
        up3 = UpSampling2D((2, 2))(conv6)
        decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)
        return decoder


# %%

autoencoder = AutoEncoder(input_img)
autoencoder.autoencoder_model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

# %%

input_data, labels = get_images('faces94')

# %%

noise_factor = 0.1
X = input_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=input_data.shape)
X = np.clip(X, 0., 1.)

# %%

epochs = 100
batch_size = 128

autoencoder.autoencoder_model.fit(X, input_data, batch_size=batch_size, epochs=epochs, validation_split=0.3, verbose=1)

# %%

encoded = autoencoder.encoder_model.predict(input_data)

# %%

df = pd.DataFrame(np.concatenate([labels, encoded], axis=1))
df.to_csv('encoded.csv', index=False)
