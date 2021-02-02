import os
import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm

from keras.layers import Input, Flatten, Embedding, multiply
from keras.layers import BatchNormalization
from keras.models import Model, Sequential
from keras.layers.core import Dense,Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import initializers

#os.environ["KERAS_BACKEND"] = "tensorflow"
np.random.seed(39)
latent_dim = 100
img_shape = (28, 28, 1)
num_classes = 10
img_dim = 784
init = initializers.RandomNormal(stddev=0.02)

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 127.5 -1.
    x_train = x_train.reshape(60000, img_dim)
    y_train = y_train.reshape(-1, 1)
    return (x_train, y_train, x_test, y_test)

def get_optimizer():
    return Adam (lr=0.0002, beta_1=0.5)

def get_generator(optimizer):

    generator = Sequential()

    generator.add(Dense(256, input_shape=(latent_dim,), kernel_initializer=init))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dense(img_dim, activation='tanh'))

    label = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(num_classes, latent_dim)(label)
    label_embedding = Flatten()(label_embedding)
    noise = Input(shape=(latent_dim,))
    model_input = multiply([noise, label_embedding])
    img = generator(model_input)

    generator = Model([noise, label], img)
    generator.summary()

    return generator

def get_discriminator(opt):
    discriminator = Sequential()
    discriminator.add(Dense(512, input_shape=(img_dim,), kernel_initializer=init))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(1, activation='sigmoid'))

    label = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(num_classes, img_dim)(label)
    label_embedding = Flatten()(label_embedding)

    img = Input(shape=(img_dim,))

    dis_input = multiply([img, label_embedding])
    validity = discriminator(dis_input)

    discriminator = Model([img, label], validity)
    discriminator.summary()

    discriminator.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'] )

    return discriminator

def get_cgan_network(discriminator, latent_dim, generator, opt):
    discriminator.trainable = False

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    validity = discriminator([generator([noise, label]), label])

    cgan = Model(inputs=[noise,label], outputs = validity)
    cgan.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])

    cgan.summary()

    return cgan

def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/%d.png' % epoch)
    plt.close()

def plot_images(e, generator, samples=10, figsize=(10, 10)):

    z = np.random.normal(loc=0, scale=1, size=(samples, latent_dim))
    labels = np.arange(0, 10).reshape(-1, 1)

    x_fake = generator.predict([z, labels])
    plt.figure(figsize=figsize)
    for k in range(samples):
        plt.subplot(2, 5, k + 1)
        plt.imshow(x_fake[k].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('cgan_images/%d.png' % e)
    plt.close()


def train(epochs=1, batch_size=128):
    x_train, y_train, _, _ = load_mnist_data()
    smooth = 0.1

    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    cgan = get_cgan_network(discriminator, latent_dim, generator, adam)

    y_dis = np.concatenate([np.ones((batch_size, 1)) * (1 - smooth), np.zeros((batch_size, 1))])
    y_gen = np.ones((batch_size, 1))

    for e in range(epochs + 1):
        for i in range(x_train.shape[0] // batch_size):
            discriminator.trainable = True
            # idx = np.random.randint(0, x_train.shape[0], size=batch_size)
            # image_batch, labels = x_train[idx], y_train[idx]

            X_batch = x_train[i * batch_size: (i + 1) * batch_size]
            real_labels = y_train[i * batch_size: (i + 1) * batch_size].reshape(-1, 1)

            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            random_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
            generated_images = generator.predict_on_batch([noise, random_labels])

            combined_labels = np.concatenate([real_labels, random_labels])
            X = np.concatenate([X_batch, generated_images])

            d_loss = discriminator.train_on_batch([X, combined_labels], y_dis)

            #discriminator.trainable = False

            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            random_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
            g_loss = cgan.train_on_batch([noise, random_labels], y_gen)
            print("%d batch %d/%d [D loss: %f, acc: %f] [G loss: %f]" % (
            e, i, x_train.shape[0] // batch_size, d_loss[0], d_loss[1], g_loss[0]))

        plot_images(e, generator, 10)


if __name__ == '__main__':
    train(300, 128)




