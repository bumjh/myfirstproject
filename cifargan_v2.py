from keras.datasets.cifar10 import load_data
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU
from keras.layers import Conv2DTranspose, Reshape
from keras.utils.vis_utils import plot_model
import numpy as np


# x_train, y_train, x_test, y_test = load_data()
# print('Train', x_train.shape, y_train.shape)
# print('Test', x_test.shape, y_test.shape)

# for i in range(49):
#     plt.subplot(7, 7, 1 + i)
#     plt.axis('off')
#     plt.imshow(x_train[i])
# plt.show()


# define the standalone discriminator model
def get_discriminator(in_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def get_generator(latent_dim):
    model = Sequential()
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((4, 4, 256)))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model


def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(0.0002, 0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def load_samples():
    (x_train, _), (_, _) = load_data()
    x_train = x_train / 127.5 - 1.
    return x_train


def generate_real_samples(dataset, n_samples):
    idx = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[idx]
    y = np.ones((n_samples, 1))
    return X, y


def generate_latent_input(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_input(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y


def save_plot(examples, epoch, n=7):
    examples = (examples + 1) / 2.0

    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i])
    # save plot to file
    filename = 'images/generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):

    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)

    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))

    save_plot(x_fake, epoch)

    filename = 'saved_model/generator_model_%03d.h5' % (epoch+1)
    g_model.save(filename)


def train(g_model, d_model, gan_model, x_train, latent_dim, epochs=200, batch_size=128):
    smooth = 0.1

    y_real = np.ones((batch_size,1))*(1-smooth)
    y_gan = np.ones((batch_size, 1))

    bat_per_epo = x_train.shape[0] // batch_size

    for i in range(epochs + 1):
        for j in range(bat_per_epo):
            d_model.trainable = True
            X_batch = x_train[i*batch_size : (i+1)*batch_size]
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, batch_size)
            d_loss = d_model.train_on_batch(np.concatenate([X_batch, X_fake]),np.concatenate([y_real, y_fake]))

            X_gan = generate_latent_input(latent_dim, batch_size)
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            if j % 50 == 0:
                print('>%d, %d/%d, d1= %.3f, d2=%.3f' % (i, j, bat_per_epo, d_loss[0], g_loss))
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, x_train, latent_dim)



dataset = load_samples()
latent_dim = 100
d_model = get_discriminator()
g_model = get_generator(latent_dim)
gan_model = define_gan(g_model, d_model)
gan_model.summary()

# plot_model(d_model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
# plot_model(g_model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
# plot_model(gan, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)


train(g_model, d_model, gan_model, dataset, latent_dim, epochs=100, batch_size=256)
