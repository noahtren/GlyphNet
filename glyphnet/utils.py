import tensorflow as tf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def visualize(symbols, glyphs, title):
    """Visualize a square of samples and their symbol labels

    Args:
        symbols: a list of strings that correspond to each glyph
        glyphs: a tensor or list of glyphs. These should be normalized from
            0 to 1 by a sigmoid activation function
        title: title of the plot
    """
    dim = len(glyphs) ** (1/2)
    assert dim == int(dim), "Number of glyphs should be square"
    dim = int(dim)
    glyphs = glyphs * 255. # make visual
    fig = make_subplots(dim, dim, subplot_titles=symbols)
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
    fig.show()
