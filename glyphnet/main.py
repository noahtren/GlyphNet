"""Train generator and discriminator to collaboratively communicate glyphs
"""

import argparse
import code

import tensorflow as tf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from glyphnet.models import make_generator_with_opt, make_discriminator_with_opt
from glyphnet.noise import random_generator_noise, random_glyphs, get_noisy_channel
from glyphnet.utils import visualize


def get_opt():
    """Use argparse library to assign options for training run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector_dim', type=int, default=32,
                        help='The size of the vector to communicate')
    parser.add_argument('--encoding', type=str, default='one-hot',
                        help="'one-hot' to make all values 0 except for one. Unique num = vector_dim" +
                             "\n'binary' to make all values either 0 or 1, randomly distributed. Unique num = vector_dim ^ 2")
    parser.add_argument('-r', type=int, default=6,
                        help='Number of upsample/downsample layers in G and D')
    parser.add_argument('--num_filters', type=int, default=8,
                        help='Number of filters to use right before and after the signal')
    parser.add_argument('-c', type=int, default=1,
                        help='Number of channels in the signal. 3 produces color images')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--no_noise', action='store_true', default=False,
                        help='Boolean flag to train with or without a noisy channel')
    parser.add_argument('--vis_frequency', type=int, default=1,
                        help='How many epochs between visualizationss')
    parser.add_argument('--debug', action='store_true', default=False, help='Toggle debug mode (prints gradient information)')
    opt = parser.parse_args()
    return opt


def make_signals(batch_size, encoding, vector_dim):
    """Generate a batch of signals and labels. Label is the same as signal
    except it has an additional 0 tagged to the end, representing "not noise"
    """
    if encoding == 'one-hot':
        indices = tf.random.uniform((batch_size,), minval=0, 
                                    maxval=vector_dim, dtype=tf.int32)
        signal = tf.one_hot(indices, dtype=tf.float32, depth=vector_dim)
        # add negative noise label (since this is a signal)
        zeros = tf.zeros((batch_size, 1), dtype=tf.float32)
        label = tf.concat((signal, zeros), axis=1)
        return signal, label
    elif encoding == 'binary':
        signal = tf.random.uniform((batch_size, vector_dim), minval=0, 
                                   maxval=2, dtype=tf.int32)
        signal = tf.cast(signal, tf.float32)
        # add negative noise label (since this is a signal)
        zeros = tf.zeros((batch_size, 1), dtype=tf.float32)
        label = tf.concat((signal, zeros), axis=1)
        return signal, label
    else:
        raise RuntimeError(f"Encoding '{encoding}' not understood")


def make_noise_labels(batch_size, vector_dim):
    """Generate message labels that represent noise (all 0 except last dimension)
    """
    zeros = tf.zeros((batch_size, vector_dim), dtype=tf.float32)
    ones = tf.ones((batch_size, 1), dtype=tf.float32)
    noise_label = tf.concat((zeros, ones), axis=1)
    return noise_label


def get_glyph_symbol(signal):
    """Get an alphanumeric symbol to associate with a given signal/glyph
    """
    if opt.encoding == 'one-hot':
        signal = tf.argmax(signal).numpy()
        if opt.vector_dim <= 26:
            return chr(signal + 97)
        else:
            return str(signal)
    elif opt.encoding == 'binary':
        return ''.join(str(int(digit.numpy())) for digit in signal)
    else:
        raise RuntimeError(f"Encoding '{opt.encoding}' not understood")


def visualize_samples(dim, title):
    """Generate a square of samples and visualize them
    """
    num_glyphs = dim ** 2
    signals, labels = make_signals(num_glyphs, opt.encoding, opt.vector_dim)
    symbols = [get_glyph_symbol(signal) for signal in signals]
    glyphs = G(signals)
    visualize(symbols, glyphs, title)


if __name__ == "__main__":
    opt = get_opt()
    glyph_size = 2 ** opt.r
    glyph_shape = [glyph_size, glyph_size, opt.c]

    G = make_generator_with_opt(opt)
    D = make_discriminator_with_opt(opt)
    G.summary()
    D.summary()

    loss_fn = None
    if opt.encoding == 'one-hot':
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    G_optim = tf.keras.optimizers.Adam(lr=0.001)
    D_optim = tf.keras.optimizers.Adam(lr=0.001)
    # noisy channel adds noise to generated glyphs
    noisy_channel = get_noisy_channel()

    for epoch in range(opt.epochs):
        random_G = make_generator_with_opt(opt)
        for step in range(opt.steps_per_epoch):

            # Train discriminator on positive example
            signals, labels = make_signals(opt.batch_size, opt.encoding, opt.vector_dim)
            glyphs = G(signals)
            # code.interact(local=locals())
            glyphs = glyphs if opt.no_noise else noisy_channel(glyphs)
            with tf.GradientTape() as tape:
                D_pred = D(glyphs)
                D_loss = loss_fn(labels, D_pred)
            grads = tape.gradient(D_loss, D.trainable_variables)
            D_optim.apply_gradients(zip(grads, D.trainable_variables))
            if opt.debug:
                print(f'D average grads: {tf.math.reduce_mean(grads[-1])}', end=' ')

            # Train discriminator on random example
            # every other step, the example is either from:
            #  * a randomly initialized generator, or
            #  * a function that produces random noise
            noise_glyphs = None
            if step % 2 == 0:
                signals, _ = make_signals(opt.batch_size, opt.encoding, opt.vector_dim)
                noise_labels = make_noise_labels(opt.batch_size, opt.vector_dim)
                noise_glyphs = random_G(signals)
            else:
                noise_glyphs = random_glyphs(opt.batch_size, glyph_shape)
            with tf.GradientTape() as tape:
                D_pred = D(noise_glyphs)
                D_fake_loss = loss_fn(noise_labels, D_pred)
            grads = tape.gradient(D_fake_loss, D.trainable_variables)
            D_optim.apply_gradients(zip(grads, D.trainable_variables))

            # Train cooperative generator
            signals, labels = make_signals(opt.batch_size, opt.encoding, opt.vector_dim)
            with tf.GradientTape() as tape:
                glyphs = G(signals)
                glyphs = glyphs if opt.no_noise else noisy_channel(glyphs)
                D_pred = D(glyphs)
                G_loss = loss_fn(labels, D_pred)
            grads = tape.gradient(G_loss, G.trainable_variables)
            G_optim.apply_gradients(zip(grads, G.trainable_variables))
            if opt.debug:
                print(f'G average grads: {tf.math.reduce_mean(grads[-1])}', end=' ')

            print(f'[{epoch+1}/{opt.epochs}] [{step+1}/{opt.steps_per_epoch}] ' +
                  f'Loss: {D_loss + G_loss + D_fake_loss}')
        if (epoch + 1) % opt.vis_frequency == 0:
            visualize_samples(3, title=f'Epoch {epoch + 1}')
