# Glyphnet

Glyphnet is an experiment in using neural networks to generate a visual language.
This project is inspired by Joel Simon's [Dimensions of Dialogue](https://www.joelsimon.net/dimensions-of-dialogue.html)
work, but all of the code here is original.

## How it works

The task is based on the Shannon-Weaver model of communication, where
the transmitter and receiver roles are played by two neural networks, G (a generator)
and D (a discriminator).

![Shannon-Weaver Model](https://i.imgur.com/0F8K9jX.png)

G turns a **message** into an image (**signal**) that the discriminator D tries to
decode back into the original message. The message is either a one-hot encoded
vector or a vector of binary values (ex. [0, 1, 1, 0...]). The two networks are trained
at the same time, like an autoencoder.

The task gets difficult because the communication channel is noisy. The transmitted image
(**signal**) may be shifted, resized, rotated, etc. so that the received signal differs
from the original signal.

Another catch is that the communication channel becomes increasingly noisy over time (this
is inspired by [Automatic Domain Randomization](https://openai.com/blog/solving-rubiks-cube/)
and [POET](https://eng.uber.com/poet-open-ended-deep-learning/)). When G and D get too 
competent in their current communication channel, the the channel becomes more noisy,
encouraging the language to evolve further.

If the two neural networks are successful and perform well at a high difficult level, then they've created
a visual language that is robust to many different kinds of noise, which means it probably looks cool. 😎️

## Early Examples

These are some examples of visual languages the models came up with after about 
a day of coding. The plot titles are either letter or numbers, representing a unique
symbol.

![](https://i.imgur.com/NNh58Nx.png)

![](https://i.imgur.com/NkSESQL.png)

## Installation

Glyphnet itself is a Python package that depends on TensorFlow 2.0, tensorflow-addons and plotly. You
can install the package and its dependencies locally with pip.

```
git clone https://github.com/noahtren/glyphnet
pip install -e .
```

## Run it

The `main.py` module is used to begin training. 
It produces a visualization of progress at the end of each epoch.

Each run can use different settings which are passed
at the command line. Run `python main.py --help` for descriptions of each setting.
