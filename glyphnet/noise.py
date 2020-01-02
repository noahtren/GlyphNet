"""Different strategies for introducing noise into the communication channel
"""

import tensorflow as tf
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
    def static(glyphs, stddev=0.01):
        glyph_shape = glyphs[0].shape
        batch_size = glyphs.shape[0]
        noise = tf.random.normal((batch_size, *glyph_shape), mean=0, stddev=stddev)
        glyphs = glyphs + noise
        return glyphs


    @staticmethod
    def translate(glyphs, minval=-2, maxval=2):
        """Shift each glyph a pixel distance from minval to maxval, using zero padding
        > For efficiency, each batch is augmented in the same way
        """
        glyph_shape = glyphs[0].shape
        batch_size = glyphs.shape[0]
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
    def resize(glyphs, minscale=0.9, maxscale=1.1):
        glyph_shape = glyphs[0].shape
        batch_size = glyphs.shape[0]
        x_scale, y_scale = tf.random.uniform([2], minval=minscale, maxval=maxscale)
        target_width = tf.cast(x_scale * glyph_shape[1], tf.int32)
        target_height = tf.cast(y_scale * glyph_shape[0], tf.int32)
        glyphs = tf.image.resize(glyphs, (target_height, target_width))
        glyphs = tf.image.resize_with_crop_or_pad(glyphs, glyph_shape[0], glyph_shape[1])
        return glyphs


def get_noisy_channel(func_names=['translate', 'resize', 'static']):
    """Return a function that adds noise to glyphs
    """
    def noise_pipeline(glyphs, funcs):
        """Apply a series of functions to glyphs, in order
        """
        for func in funcs:
            glyphs = func(glyphs)
        return glyphs
    funcs = []
    for func_name in func_names:
        assert func_name in dir(Differentiable_Augment), f"Function '{func_name}' doesn't exist"
        funcs.append(getattr(Differentiable_Augment, func_name))
    return lambda glyphs: noise_pipeline(glyphs, funcs)


# preview image augmentation
if __name__ == "__main__":
    symbols = ['random'] * 9
    glyphs = random_glyphs(9, [16, 16, 1])
    noisy_chanel = get_noisy_channel()
    new_glyphs = noisy_chanel(glyphs)
    visualize(symbols, glyphs, 'Before Augmentation')
    visualize(symbols, new_glyphs, 'After Augmentation')
