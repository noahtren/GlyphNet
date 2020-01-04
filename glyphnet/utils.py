import tensorflow as tf

import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def get_glyph_symbol(message, encoding, vector_dim):
    """Get an alphanumeric symbol to associate with a given message/glyph
    """
    if encoding == 'one-hot':
        message = tf.argmax(message).numpy()
        if vector_dim <= 26:
            return chr(message + 97)
        else:
            return str(message)
    elif encoding == 'binary':
        return ''.join(str(int(digit.numpy())) for digit in message)
    else:
        raise RuntimeError(f"Encoding '{encoding}' not understood")


def visualize(symbols, glyphs, title, get_fig=False, use_titles=True):
    """Visualize a square of samples and their symbol labels

    Args:
        symbols: a list of strings that correspond to each glyph
        glyphs: a tensor or list of glyphs. These should be normalized from
            0 to 1 by a sigmoid activation function
        title: title of the plot
        get_fig: if true, return the Plotly figure object instead of visualizing immediately
    """
    dim = len(glyphs) ** (1/2)
    assert dim == int(dim), "Number of glyphs should be square"
    dim = int(dim)
    glyphs = np.array(glyphs * 255.).astype('uint8') # make visual
    fig = make_subplots(dim, dim, subplot_titles=symbols if use_titles else None, horizontal_spacing=0.01, vertical_spacing=0.01)
    row_num = 1; col_num = 1
    for glyph in glyphs:
        if glyph.shape[2] == 1:
            glyph = tf.broadcast_to(glyph, (glyph.shape[0], glyph.shape[1], 3))
        fig.add_trace(go.Image(z=glyph), row=row_num, col=col_num)
        if col_num == dim:
            row_num += 1
            col_num = 1
        else:
            col_num += 1
    fig.update_layout(title_text=title)
    if get_fig:
        return fig
    else:
        fig.show()
