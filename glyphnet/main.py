"""Train generator and discriminator to collaboratively communicate glyphs
"""

import argparse

import tensorflow as tf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from glyphnet.models import generator, discriminator


def get_opt():
    """Use argparse library to assign options for training run.

    Defaults:
        vector_dim = 32 (grammar size of 32)
        r = 4 (4 upsample/downsample steps)
        num_filters = 8 (8 filters before and after glyph)
        c = 1 (monochrome glyph)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector_dim', default=26)
    parser.add_argument('-r', default=4)
    parser.add_argument('--num_filters', default=8)
    parser.add_argument('-c', default=1)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--epochs', default=30)
    parser.add_argument('--steps_per_epoch', default=100)
    opt = parser.parse_args()
    return opt


def make_signal(batch_size, get_indices=False):
    """Generate a batch of signals

    Args:
        batch_size: number of signals to generate
        get_original: whether or not to get the indices as well as the one-hot encoded
            signal
    """
    indices = tf.random.uniform((batch_size,), minval=0, 
                               maxval=opt.vector_dim, dtype=tf.int32)
    signal = tf.one_hot(indices, depth=opt.vector_dim)
    if get_indices:
        return signal, indices
    else:
        return signal


def get_glyph_symbol(index):
    """Get an alphanumeric symbol to associate with a given glyph index
    """
    if opt.vector_dim <= 26:
        return chr(index + 97)
    else:
        return index


def visualize_samples(sqrt_samples, title):
    """Generates a square of samples to visualize all at once
    """
    num_samples = sqrt_samples ** 2
    signal, indices = make_signal(num_samples, get_indices=True)
    glyphs = G(signal)
    glyphs = tf.nn.sigmoid(glyphs) * 255.
    glyph_size = 2 ** opt.r
    subplot_titles = [get_glyph_symbol(index) for index in indices]
    fig = make_subplots(sqrt_samples, sqrt_samples, subplot_titles=subplot_titles)
    row_num = 1; col_num = 1
    for glyph in glyphs:
        if opt.c == 1:
            glyph = tf.broadcast_to(glyph, (glyph_size, glyph_size, 3))
        fig.add_trace(go.Image(z=glyph), row=row_num, col=col_num)
        if col_num == sqrt_samples:
            row_num += 1
            col_num = 1
        else:
            col_num += 1
    fig.update_layout(title_text=title)
    fig.show()


if __name__ == "__main__":
    opt = get_opt()

    G = generator(opt.vector_dim, opt.r, opt.num_filters, opt.c)
    D = discriminator(opt.vector_dim, opt.r, opt.num_filters, opt.c)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True) # one hot prediction

    G_optim = tf.keras.optimizers.Adam()
    D_optim = tf.keras.optimizers.Adam()

    for epoch in range(opt.epochs):
        for step in range(opt.steps_per_epoch):
            # Train discriminator
            signal = make_signal(opt.batch_size)
            glyphs = G(signal)
            with tf.GradientTape() as tape:
                D_pred = D(glyphs)
                D_loss = loss_fn(signal, D_pred)
            grads = tape.gradient(D_loss, D.trainable_variables)
            D_optim.apply_gradients(zip(grads, D.trainable_variables))

            # Train generator
            signal = make_signal(opt.batch_size)
            with tf.GradientTape() as tape:
                glyphs = G(signal)
                D_pred = D(glyphs)
                G_loss = loss_fn(signal, D_pred)
            grads = tape.gradient(G_loss, G.trainable_variables)
            G_optim.apply_gradients(zip(grads, G.trainable_variables))

            print(f'[{epoch+1}/{opt.epochs}] [{step+1}/{opt.steps_per_epoch}] Loss: {D_loss + G_loss}')
        if (epoch + 1) % 10 == 0:
            visualize_samples(3, title=f'Epoch {epoch + 1}')


