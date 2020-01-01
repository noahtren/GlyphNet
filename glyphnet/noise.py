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
    Each of these functions operates on a single glyph and can be applied in parallel
    with the tf.data.Dataset API.
    """

    @staticmethod
    def static(glyph, glyph_shape):
        noise = tf.random.normal((*glyph_shape), mean=0, stddev=0.1)
        glyph = glyph + noise
        return glyph


    @staticmethod
    @tf.function
    def translate(glyph, minval=-5, maxval=5):
        """Shift each glyph a pixel distance from minval to maxval, using zero padding
        """
        glyph_shape = glyph.shape
        shifts = tf.random.uniform([2], minval=minval, maxval=maxval + 1, dtype=tf.int32)
        shift_x = shifts[0]; shift_y = shifts[1]
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
        return glyph


    @staticmethod
    @tf.function
    def resize(glyph, minscale=0.8, maxscale=1.2):
        glyph_shape = glyph.shape
        scales = tf.random.uniform([2], minval=minscale, maxval=maxscale)
        x_scale = scales[0]; y_scale = scales[1]
        target_width = tf.cast(x_scale * glyph_shape[1], tf.int32)
        target_height = tf.cast(y_scale * glyph_shape[0], tf.int32)
        glyph = tf.image.resize(glyph, (target_height, target_width))
        glyph = tf.image.resize_with_crop_or_pad(glyph, glyph_shape[0], glyph_shape[1])
        return glyph


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


def get_noise_dataset(glyphs, funcs, batch_size):
    noise_ds = tf.data.Dataset.from_tensor_slices(glyphs)
    for func in funcs:
        noise_ds = noise_ds.map(func, num_parallel_calls=batch_size)
    return noise_ds


def experimental_noisy_channel(func_names=['resize']):
    funcs = []
    for func_name in func_names:
        assert func_name in dir(Differentiable_Augment), f"Function '{func_name}' doesn't exist"
        funcs.append(getattr(Differentiable_Augment, func_name))
    def get_new_glyphs(glyphs):
        batch_size = glyphs.shape[0]
        noise_ds = get_noise_dataset(glyphs, funcs, batch_size)
        for new_glyphs in noise_ds.batch(batch_size):
            return new_glyphs
    return lambda glyphs: get_new_glyphs(glyphs)


# preview image augmentation
if __name__ == "__main__":
    symbols = ['random'] * 9
    glyphs = random_glyphs(9, [16, 16, 1])
    noisy_chanel = experimental_noisy_channel(func_names=['resize', 'translate'])
    new_glyphs = noisy_chanel(glyphs)
    visualize(symbols, glyphs, 'Before Augmentation')
    visualize(symbols, new_glyphs, 'After Augmentation')
