"""Different strategies for introducing noise into the communication channel
"""

import tensorflow as tf

from glyphnet.models import make_generator_with_opt


def random_generator_noise(signals, opt):
    """Generate glyphs with a randomly initialized generator
    """
    random_G = make_generator_with_opt(opt)
    noise_glyphs = random_G(signals)
    return noise_glyphs


def random_glyphs(batch_size, glyph_size, c):
    """Normally distributed noise images
    """
    noise_glyphs = tf.random.normal((batch_size, glyph_size, glyph_size, c), mean=0, stddev=1)
    noise_glyphs = tf.nn.sigmoid(noise_glyphs)
    return noise_glyphs


class Channel_Noise:
    """Functions that add noise to a tensor of glyphs
    """

    @staticmethod
    def static(glyphs, glyph_size, c):
        glyphs = glyphs + tf.random.normal((glyph_size, glyph_size, c), mean=0.5, stddev=0.01)
        return glyphs

def get_noisy_channel(func_names=['static']):
    """Return a function that adds noise to glyphs
    """
    def noise_pipeline(glyphs, glyph_size, c, funcs):
        """Apply a series of functions to glyphs, in order
        """
        for func in funcs:
            glyphs = func(glyphs, glyph_size, c)
        return glyphs

    funcs = []
    for func_name in func_names:
        assert '__' not in func_name
        assert func_name in dir(Channel_Noise)
        funcs.append(getattr(Channel_Noise, func_name))

    return lambda glyphs, glyph_size, c: noise_pipeline(glyphs, glyph_size, c, funcs)
