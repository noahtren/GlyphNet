"""Train generator and discriminator to collaboratively communicate glyphs
"""

import argparse

import tensorflow as tf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from glyphnet.models import make_generator_with_opt, make_discriminator_with_opt
from glyphnet.noise import random_generator_noise, random_glyphs, experimental_noisy_channel
from glyphnet.utils import visualize


def get_opt():
    """Use argparse library to assign options for training run.

    Descriptions of options:
        * vector_dim is the size of the vector to communicate
        * encoding describes the kinds of values that the vector can contain:
            one-hot: all values are 0 except 1
            binary: all values are either 0 or 1, uniformly distributed
        * r ^ 2 is size of images
        * num_filters is the number of filters before and after image
        * c is number of channels
    TODO: set as help in CLI
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector_dim', type=int, default=26)
    parser.add_argument('--encoding', type=str, default='one-hot')
    parser.add_argument('-r', type=int, default=4)
    parser.add_argument('--num_filters', type=int, default=8)
    parser.add_argument('-c', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--no_noise', action='store_true', default=False)
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
    zeros = tf.zeros((batch_size, vector_dim), dtype=tf.float32)
    ones = tf.ones((batch_size, 1), dtype=tf.float32)
    noise_label = tf.concat((zeros, ones), axis=1)
    return noise_label


def get_glyph_symbol(signal):
    """Get an alphanumeric symbol to associate with a given signal/glyph
    """
    if opt.encoding == 'one-hot':
        signal = tf.argmax(signal)
        if opt.vector_dim <= 26:
            return chr(signal + 97)
        else:
            return signal
    elif opt.encoding == 'binary':
        return ''.join(str(int(digit.numpy())) for digit in signal)
    else:
        raise RuntimeError(f"Encoding '{opt.encoding}' not understood")


def visualize_samples(dim, title):
    """Generate a square of samples and visualize them
    """
    num_glyphs = dim ** 2
    signals = make_signals(num_glyphs, opt.encoding, opt.vector_dim)
    symbols = [get_glyph_symbol(signal) for signal in signals]
    glyphs = G(signals)
    visualize(symbols, glyphs, title)


if __name__ == "__main__":
    opt = get_opt()
    glyph_size = 2 ** opt.r
    glyph_shape = [glyph_size, glyph_size, opt.c]

    G = make_generator_with_opt(opt)
    D = make_discriminator_with_opt(opt)

    loss_fn = None
    if opt.encoding == 'one-hot':
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    G_optim = tf.keras.optimizers.Adam()
    D_optim = tf.keras.optimizers.Adam()
    # noisy channel adds noise to generated glyphs
    noisy_channel = experimental_noisy_channel()

    for epoch in range(opt.epochs):
        random_G = make_generator_with_opt(opt)
        for step in range(opt.steps_per_epoch):

            # Train discriminator on positive example
            signals, labels = make_signals(opt.batch_size, opt.encoding, opt.vector_dim)
            glyphs = G(signals)
            glyphs = glyphs if opt.no_noise else noisy_channel(glyphs)
            with tf.GradientTape() as tape:
                D_pred = D(glyphs)
                D_loss = loss_fn(labels, D_pred)
            grads = tape.gradient(D_loss, D.trainable_variables)
            D_optim.apply_gradients(zip(grads, D.trainable_variables))

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

            print(f'[{epoch+1}/{opt.epochs}] [{step+1}/{opt.steps_per_epoch}] ' +
                  f'Loss: {D_loss + G_loss + D_fake_loss}')
        if (epoch + 1) % 10 == 0:
            visualize_samples(3, title=f'Epoch {epoch + 1}')
