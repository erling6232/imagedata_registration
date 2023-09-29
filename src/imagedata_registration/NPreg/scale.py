"""
SCALE(IM) Scaling.
SCALE(IM) scales the image IM between 0 and 1.
SCALE(IM,LOW,HIGH) scales the image IM between LOW and HIGH.
"""

import numpy as np


def scale(im, low=0., high=1.):
    minscale = np.amin(im)
    im = im - minscale

    maxscale = np.amax(im)
    if maxscale == 0:
        raise ValueError("Empty image.")
    im = im / maxscale

    # scale to high and low
    im = im * (high - low)
    im = im + low
    return im, minscale, maxscale
