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
    return tf.random.normal((glyph_size, glyph_size, c), mean=0, stddev=5.0)
