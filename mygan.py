import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense,Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import initializers

#os.environ["KERAS_BACKEND"] = "tensorflow"
np.random.seed(39)
random_dim = 100

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 127.5 -1.
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)

def get_optimizer():
    return Adam (lr=0.0002, beta_1=0.5)

def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(784, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    generator.summary()

    return generator

def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    discriminator.summary()
    return discriminator


def get_gan_network(discriminator, random_dim, generator, optimizer):
    discriminator.trainable = False

    gan_input = Input(shape=(random_dim,))
    x = generator(gan_input)

    gan_output = discriminator(x)

    gan = Model(inputs=gan_input, outputs = gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    gan.summary()

    return gan


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('my_images1/%d.png' % epoch)
    plt.close()

def train(epochs=1, batch_size=128):
    x_train, _, _, _ = load_mnist_data()

    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(epochs+1):
        image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

        noise = np.random.normal(0, 1, size= [batch_size, random_dim])
        generated_images = generator.predict(noise)
        X = np.concatenate([image_batch, generated_images])
        y_dis = np.concatenate([np.ones(batch_size)*0.99, np.zeros(batch_size)])

        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X, y_dis)

        noise = np.random.normal(0, 1, size= [batch_size, random_dim])
        y_gen = np.ones(batch_size)
        #discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y_gen)
        
        if e % 100 ==0:
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (e, d_loss[0], 100 * d_loss[1], g_loss))
        
        if e % 10000 == 0:
            plot_generated_images(e, generator)

if __name__ == '__main__':
    train(10000, 128)
    
