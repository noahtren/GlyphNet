"""Different strategies for introducing noise into the communication channel
"""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

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


class Differentiable_Augment:
    """Collection of differentiable augmentation functions implemented in TF
    """

    @staticmethod
    def static(glyphs, DIFFICULTY):
        STATIC_STDDEVS = {
            0: 0.00,
            1: 0.01,
            2: 0.02,
            3: 0.03,
            4: 0.04,
            5: 0.05
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
            1: 0.05,
            2: 0.075,
            3: 0.1,
            4: 0.125,
            5: 0.15,
        }
        glyph_shape = glyphs[0].shape
        batch_size = glyphs.shape[0]
        max_shift_percent = SHIFT_PERCENTS[DIFFICULTY]
        average_glyph_dim = (glyph_shape[0] + glyph_shape[1]) / 2
        minval = int(max_shift_percent * average_glyph_dim * -1)
        maxval = int(max_shift_percent * average_glyph_dim)
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
        }
        glyph_shape = glyphs[0].shape
        batch_size = glyphs.shape[0]
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
            1: 1 * pi / 20,
            2: 2 * pi / 20,
            3: 3 * pi / 20,
            4: 4 * pi / 20,
            5: 5 * pi / 20,
        }
        min_angle = RADIANS[DIFFICULTY] * -1
        max_angle = RADIANS[DIFFICULTY]
        glyph_shape = glyphs[0].shape
        batch_size = glyphs.shape[0]
        angles = tf.random.uniform([batch_size], minval=min_angle, maxval=max_angle)
        glyphs = tfa.image.rotate(glyphs, angles, interpolation='BILINEAR')
        return glyphs


def get_noisy_channel(func_names=['translate', 'resize', 'rotate', 'static']):
    """Return a function that adds noise to glyphs
    """
    def noise_pipeline(glyphs, funcs, DIFFICULTY):
        """Apply a series of functions to glyphs, in order
        """
        for func in funcs:
            glyphs = func(glyphs, DIFFICULTY)
        return glyphs
    funcs = []
    for func_name in func_names:
        assert func_name in dir(Differentiable_Augment), f"Function '{func_name}' doesn't exist"
        funcs.append(getattr(Differentiable_Augment, func_name))
    return lambda glyphs, DIFFICULTY: noise_pipeline(glyphs, funcs, DIFFICULTY)


# preview image augmentation
if __name__ == "__main__":
    DIFFICULTY = 4
    symbols = ['random'] * 9
    glyphs = random_glyphs(9, [16, 16, 1])
    noisy_chanel = get_noisy_channel()
    new_glyphs = noisy_chanel(glyphs, DIFFICULTY)
    visualize(symbols, glyphs, 'Before Augmentation')
    visualize(symbols, new_glyphs, f'After Augmentation (difficulty = {DIFFICULTY})')
