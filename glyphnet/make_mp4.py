"""Generate N messages in order and visualize signals after each epoch
"""

import os

import tensorflow as tf
import plotly.graph_objects as go
import imageio
from tqdm import tqdm

from glyphnet.utils import visualize, get_glyph_symbol
from glyphnet.models import swish

RUN_NAME = 'color_noise'
ENCODING = 'one-hot'
VECTOR_DIM = 128
PREVIEW = False
message_dim = 64 # size of image
num_message_dim = 6 # size of display
num_messages = num_message_dim ** 2

def make_first_n_messages(n, vector_dim, encoding):
    if encoding == 'one-hot':
        messages = list(range(n))
        messages = tf.one_hot(messages, vector_dim)
        return messages
    else:
        raise NotImplementedError


def write_images():
    """Load each checkpoint in order and generate a visualization for each one
    """
    G_names = sorted(os.listdir(os.path.join('checkpoints', RUN_NAME, 'G')))
    D_names = sorted(os.listdir(os.path.join('checkpoints', RUN_NAME, 'D')))
    if D_names == []:
        D_names = [None] * len(G_names)

    os.makedirs(os.path.join('images', RUN_NAME), exist_ok=True)
    custom_objects = {'swish': swish}


    for i, (G_name, D_name) in tqdm(enumerate(zip(G_names, D_names))):
        G = tf.keras.models.load_model(os.path.join('checkpoints', RUN_NAME, 'G', G_name), custom_objects=custom_objects, compile=False)
        # D = tf.keras.models.load_model(os.path.join('checkpoints', RUN_NAME, 'D', D_name), custom_objects=custom_objects, compile=False)
        messages = make_first_n_messages(num_messages, VECTOR_DIM, ENCODING)
        symbols = [get_glyph_symbol(message, ENCODING, VECTOR_DIM) for message in messages]
        glyphs = G(messages)
        if PREVIEW:
            visualize(symbols, glyphs, f'Epoch {i + 1}', get_fig=False, use_titles=False)
        else:
            fig = visualize(symbols, glyphs, f'Epoch {i + 1}', get_fig=True, use_titles=False)
            # formatting
            margin = 20
            top_margin = 40
            fig.update_layout(width=(message_dim * num_message_dim * 1.1) + margin * 2,
                            height=(message_dim * num_message_dim * 1.1) + margin + top_margin,
                            margin=go.layout.Margin(l=margin,
                                                    r=margin,
                                                    b=margin,
                                                    t=top_margin,
                                                    pad=0))
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            fig.write_image(os.path.join('images', RUN_NAME, f'{i+1:03d}.png'))


def make_mp4():
    """Gather all images and generate an mp4
    """
    image_paths = sorted(os.listdir(os.path.join('images', RUN_NAME)))
    writer = imageio.get_writer(f'{RUN_NAME}.mp4')
    for image_path in image_paths:
        image = imageio.imread(os.path.join('images', RUN_NAME, image_path))
        writer.append_data(image)
    writer.close()


if __name__ == "__main__":
    if os.path.exists(os.path.join('images', RUN_NAME)):
        print("Inferring that images have already been generated. Making mp4...")
        make_mp4()
    else:
        print("Writing images first...")
        write_images()
