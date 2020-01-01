"""Different strategies for introducing noise into the communication channel
"""

import tensorflow as tf
import numpy as np

import imgaug
from imgaug import augmenters as iaa
from imgaug.parameters import Normal
from imgaug.augmentables.batches import Batch

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
    """Collection of differentiable augmentation functions to include in TF graph.
    """

    @staticmethod
    def static(glyphs, glyph_shape):
        batch_size = glyphs.shape[0]
        noise = tf.random.normal((batch_size, *glyph_shape), mean=0, stddev=0.1)
        glyphs = glyphs + noise
        return glyphs


    @staticmethod
    def translate(glyphs, glyph_shape, minval=-5, maxval=5):
        """Shift each glyph a pixel distance from minval to maxval, using zero padding
        """
        new_glyphs = []
        for glyph in glyphs:
            shift_x, shift_y = tf.random.uniform([2], minval=minval, maxval=maxval + 1, dtype=tf.int32)
            if shift_x != 0:
                zeros = tf.zeros((glyph_shape[0], abs(shift_x), glyph_shape[2]), dtype=tf.float32)
                if shift_x > 0: # shift right
                    chunk = glyph[:, :-shift_x, :]
                    glyph = tf.concat((zeros, chunk), axis=1)
                else: # shift left
                    shift_x = abs(shift_x)
                    chunk = glyph[:, shift_x:, :]
                    glyph = tf.concat((chunk, zeros), axis=1)
            if shift_y != 0:
                zeros = tf.zeros((abs(shift_y), glyph_shape[1], glyph_shape[2]), dtype=tf.float32)
                if shift_y > 0: # shift up
                    chunk = glyph[:-shift_y]
                    glyph = tf.concat((zeros, chunk), axis=0)
                else: # shift down
                    shift_y = abs(shift_y)
                    chunk = glyph[shift_y:]
                    glyph = tf.concat((chunk, zeros), axis=0)
            new_glyphs.append(glyph)
        new_glyphs = tf.stack(new_glyphs)
        return new_glyphs
    

    @staticmethod
    def resize(glyphs, glyph_shape, minscale=0.8, maxscale=1.2):
        new_glyphs = []
        for glyph in glyphs:
            x_scale, y_scale = tf.random.uniform([2], minval=minscale, maxval=maxscale)
            target_width = tf.cast(x_scale * glyph_shape[1], tf.int32)
            target_height = tf.cast(y_scale * glyph_shape[0], tf.int32)
            glyph = tf.image.resize(glyph, (target_height, target_width))
            glyph = tf.image.resize_with_crop_or_pad(glyph, glyph_shape[0], glyph_shape[1])
            new_glyphs.append(glyph)
        new_glyphs = tf.stack(new_glyphs)
        return new_glyphs


def get_noisy_channel(func_names=['translate', 'resize']):
    """Return a function that adds noise to glyphs
    """
    def noise_pipeline(glyphs, glyph_shape, funcs):
        """Apply a series of functions to glyphs, in order
        """
        for func in funcs:
            glyphs = func(glyphs, glyph_shape)
        return glyphs
    funcs = []
    for func_name in func_names:
        assert func_name in dir(Differentiable_Augment), f"Function '{func_name}' doesn't exist"
        funcs.append(getattr(Differentiable_Augment, func_name))
    return lambda glyphs, glyph_shape: noise_pipeline(glyphs, glyph_shape, funcs)


# preview image augmentation
if __name__ == "__main__":
    symbols = ['random'] * 9
    glyphs = random_glyphs(9, [16, 16, 1])
    noisy_chanel = get_noisy_channel(func_names=['translate', 'resize'])
    new_glyphs = noisy_chanel(glyphs, glyphs[0].shape)
    visualize(symbols, glyphs, 'Before Augmentation')
    visualize(symbols, new_glyphs, 'After Augmentation')
