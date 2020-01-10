"""Different strategies for introducing noise into the communication channel
"""

import random

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import plotly.express as px

from glyphnet.models import make_generator_with_opt
from glyphnet.utils import visualize


def random_generator_noise(signals, opt):
    """Generate glyphs with a randomly initialized generator
    """
    random_G = make_generator_with_opt(opt)
    noise_glyphs = random_G(signals)
    return noise_glyphs


def random_glyphs(batch_size, glyph_shape):
    """Normally distributed noise images
    """
    noise_glyphs = tf.random.normal((batch_size, *glyph_shape), mean=0, stddev=1, dtype=tf.float32)
    noise_glyphs = tf.nn.sigmoid(noise_glyphs)
    return noise_glyphs


def gaussian_k(height, width, y, x, sigma, normalized=True):
    """Make a square gaussian kernel centered at (x, y) with sigma as standard deviation.

    Returns:
        A 2D array of size [height, width] with a Gaussian kernel centered at (x, y)
    """
    # cast arguments used in calculations
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    sigma = tf.cast(sigma, tf.float32)
    # create indices
    xs = tf.range(0, width, delta=1., dtype=tf.float32)
    ys = tf.range(0, height, delta=1., dtype=tf.float32)
    ys = tf.expand_dims(ys, 1)
    # apply gaussian function to indices based on distance from x, y
    gaussian = tf.math.exp(-((xs - x)**2 + (ys - y)**2) / (2 * (sigma**2)))
    if normalized:
        gaussian = gaussian / tf.math.reduce_sum(gaussian) # all values will sum to 1
    return gaussian


class Differentiable_Augment:
    """Collection of differentiable augmentation functions implemented in TF
    """

    @staticmethod
    def static(glyphs, DIFFICULTY):
        STATIC_STDDEVS = {
            0: 0.00,
            1: 0.02,
            2: 0.04,
            3: 0.06,
            4: 0.08,
            5: 0.10,
            6: 0.13,
            7: 0.16,
            8: 0.18,
            9: 0.2,
            10: 0.21,
            11: 0.22,
            12: 0.23,
            13: 0.24,
            14: 0.25,
            15: 0.26,
        }
        glyph_shape = glyphs[0].shape
        batch_size = glyphs.shape[0]
        stddev = STATIC_STDDEVS[DIFFICULTY]
        noise = tf.random.normal((batch_size, *glyph_shape), mean=0, stddev=stddev)
        glyphs = glyphs + noise
        return glyphs


    @staticmethod
    def translate(glyphs, DIFFICULTY):
        """Shift each glyph a pixel distance from minval to maxval, using zero padding
        > For efficiency, each batch is augmented in the same way
        """
        SHIFT_PERCENTS = {
            0: 0.00,
            1: 0.025,
            2: 0.05,
            3: 0.07,
            4: 0.09,
            5: 0.1,
            6: 0.11,
            7: 0.12,
            8: 0.13,
            9: 0.14,
            10: 0.15,
            11: 0.16,
            12: 0.17,
            13: 0.18,
            14: 0.19,
            15: 0.20
        }
        glyph_shape = glyphs[0].shape
        batch_size = glyphs.shape[0]
        max_shift_percent = SHIFT_PERCENTS[DIFFICULTY]
        average_glyph_dim = (glyph_shape[0] + glyph_shape[1]) / 2
        minval = int(round(max_shift_percent * average_glyph_dim * -1))
        maxval = int(round(max_shift_percent * average_glyph_dim))
        shift_x, shift_y = tf.random.uniform([2], minval=minval, maxval=maxval + 1, dtype=tf.int32)
        if shift_x != 0:
            zeros = tf.zeros((batch_size, glyph_shape[0], abs(shift_x), glyph_shape[2]), dtype=tf.float32)
            if shift_x > 0: # shift right
                chunk = glyphs[:, :, :-shift_x]
                glyphs = tf.concat((zeros, chunk), axis=2)
            else: # shift left
                shift_x = abs(shift_x)
                chunk = glyphs[:, :, shift_x:]
                glyphs = tf.concat((chunk, zeros), axis=2)
        if shift_y != 0:
            zeros = tf.zeros((batch_size, abs(shift_y), glyph_shape[1], glyph_shape[2]), dtype=tf.float32)
            if shift_y > 0: # shift up
                chunk = glyphs[:, :-shift_y]
                glyphs = tf.concat((zeros, chunk), axis=1)
            else: # shift down
                shift_y = abs(shift_y)
                chunk = glyphs[:, shift_y:]
                glyphs = tf.concat((chunk, zeros), axis=1)
        return glyphs
    

    @staticmethod
    def resize(glyphs, DIFFICULTY):
        RESIZE_SCALES = {
            0: [1, 1],
            1: [0.9, 1.1],
            2: [0.85, 1.15],
            3: [0.8, 1.2],
            4: [0.75, 1.25],
            5: [0.7, 1.3],
            6: [0.65, 1.325],
            7: [0.6, 1.35],
            8: [0.55, 1.375],
            9: [0.50, 1.4],
            10: [0.48, 1.42],
            11: [0.46, 1.44],
            12: [0.44, 1.46],
            13: [0.42, 1.48],
            14: [0.4, 1.5],
            15: [0.4, 1.5],
        }
        glyph_shape = glyphs[0].shape
        # batch_size = glyphs.shape[0]
        minscale, maxscale = RESIZE_SCALES[DIFFICULTY]
        x_scale, y_scale = tf.random.uniform([2], minval=minscale, maxval=maxscale)
        target_width = tf.cast(x_scale * glyph_shape[1], tf.int32)
        target_height = tf.cast(y_scale * glyph_shape[0], tf.int32)
        glyphs = tf.image.resize(glyphs, (target_height, target_width))
        glyphs = tf.image.resize_with_crop_or_pad(glyphs, glyph_shape[0], glyph_shape[1])
        return glyphs
    

    @staticmethod
    def rotate(glyphs, DIFFICULTY):
        pi = 3.14159265
        RADIANS = {
            0: 0,
            1: 1 * pi / 30,
            2: 2 * pi / 30,
            3: 3 * pi / 30,
            4: 4 * pi / 30,
            5: 5 * pi / 30,
            6: 6 * pi / 30,
            7: 7 * pi / 30,
            8: 8 * pi / 30,
            9: 9 * pi / 30,
            10: 9.5 * pi / 30,
            11: 10 * pi / 30,
            12: 10.5 * pi / 30,
            13: 11 * pi / 30,
            14: 11.5 * pi / 30,
            15: 12 * pi / 30,
        }
        min_angle = RADIANS[DIFFICULTY] * -1
        max_angle = RADIANS[DIFFICULTY]
        # glyph_shape = glyphs[0].shape
        batch_size = glyphs.shape[0]
        angles = tf.random.uniform([batch_size], minval=min_angle, maxval=max_angle)
        glyphs = tfa.image.rotate(glyphs, angles, interpolation='BILINEAR')
        return glyphs


    @staticmethod
    def blur(glyphs, DIFFICULTY):
        STDDEVS = {
            0: 0.01,
            1: 0.2,
            2: 0.4,
            3: 0.6,
            4: 0.8,
            5: 0.9,
            6: 1.0,
            7: 1.25,
            8: 1.5,
            9: 1.6,
            10: 1.7,
            11: 1.8,
            12: 1.9,
            13: 2.0,
            14: 2.1,
            15: 2.2
        }
        glyph_shape = glyphs[0].shape
        c = glyph_shape[2]
        stddev = STDDEVS[DIFFICULTY]
        gauss_kernel = gaussian_k(7, 7, 3, 3, stddev)

        # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        
        # Convolve.
        out_channels = []
        for c_ in range(c):
            in_channel = tf.expand_dims(glyphs[:, :, :, c_], -1)
            out_channel = tf.nn.conv2d(in_channel, gauss_kernel, strides=1, padding="SAME")
            out_channel = tf.squeeze(out_channel)
            out_channels.append(out_channel)
        glyphs = out_channels[0] if len(out_channels) == 1 else tf.stack(out_channels, axis=-1)
        return glyphs


def random_augmentation(glyphs):
    """Apply a set of additional random augmentations that work well with single-channel glyphs.
    These include calculating the inverse of the channel, doing a harder sigmoid activation, and
    convolving the channel with a randomly initialized convolutional kernel.
    """
    if tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32) == 1:
        # hard sigmoid
        glyphs = tf.nn.sigmoid((glyphs - 0.5) * 30.)
    if glyphs.shape[3] == 1:
        if tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32) == 1:
            # inverse function
            glyphs = 1 - glyphs
        if tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32) == 1:
            # convolve with a random kernel
            kernel = tf.random.normal((3, 3), mean=0, stddev=1)
            kernel = kernel / tf.math.reduce_sum(kernel)
            kernel = kernel[:, :, tf.newaxis, tf.newaxis]
            glyphs = tf.nn.conv2d(glyphs, kernel, strides=1, padding="SAME")
    return glyphs


def get_noisy_channel(func_names=['static', 'blur', 'resize', 'translate', 'rotate']):
    """Return a function that adds noise to glyphs
    """
    def noise_pipeline(glyphs, funcs, DIFFICULTY, randomize=False):
        """Apply a series of functions to glyphs, in order
        """
        if DIFFICULTY == 0:
            return glyphs
        else:
            for func in funcs:
                this_difficulty = None
                if randomize:
                    this_difficulty = random.choice(list(range(DIFFICULTY + 1)))
                else:
                    this_difficulty = DIFFICULTY
                if this_difficulty != 0:
                    glyphs = func(glyphs, this_difficulty)
            glyphs = random_augmentation(glyphs)
            return glyphs
    funcs = []
    for func_name in func_names:
        assert func_name in dir(Differentiable_Augment), f"Function '{func_name}' doesn't exist"
        funcs.append(getattr(Differentiable_Augment, func_name))
    return lambda glyphs, DIFFICULTY, randomize=False: noise_pipeline(glyphs, funcs, DIFFICULTY, randomize)


# preview image augmentation
if __name__ == "__main__":
    symbols = ['random'] * 9
    glyphs = random_glyphs(9, [64, 64, 3])
    noisy_chanel = get_noisy_channel()
    visualize(symbols, glyphs, 'Before Augmentation')
    for DIFFICULTY in range(10):
        new_glyphs = noisy_chanel(glyphs, DIFFICULTY)
        visualize(symbols, new_glyphs, f'After Augmentation (difficulty = {DIFFICULTY})')
