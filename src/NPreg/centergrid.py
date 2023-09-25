"""CENTERGRID Find a centered grid of dimension DIM and voxelsize H."""

import numpy as np


def centergrid(dim, h):
    # Return x as dict(np.float32)

    assert issubclass(type(dim), np.ndarray), 'dim must be ndarray'
    ndim = len(dim)
    # center of image in matrix coordinates
    mid = (dim + 1) / 2.

    # natural coordinates
    # x = cell(ndim,1)
    x = {}
    for i in range(ndim):
        x[i] = np.arange(-mid[i] + 1, mid[i], h[i], dtype=float)

    if ndim == 2:
        x['row'] = x[0]
        x['column'] = x[1]
        x['size'] = 2
        x[0], x[1] = np.meshgrid(x[0], x[1], indexing='ij')
    elif ndim == 3:
        # x[0], x[1], x[2] = np.mgrid[x[0],x[1],x[2]]
        x['slice'] = x[0]
        x['row'] = x[1]
        x['column'] = x[2]
        x['size'] = 3
        x[0], x[1], x[2] = np.meshgrid(x[0], x[1], x[2], indexing='ij')
    elif ndim == 4:
        x['tag'] = x[0]
        x['slice'] = x[1]
        x['row'] = x[2]
        x['column'] = x[3]
        x['size'] = 4
        x[0], x[1], x[2], x[3] = np.meshgrid(x[0], x[1], x[2], x[3], indexing='ij')
    else:
        raise ValueError('Wrong dimension %d' % ndim)

    # max and min
    maxx = np.empty(ndim)
    maxx.fill(np.nan)
    minx = np.empty(ndim)
    minx.fill(np.nan)

    for j in range(ndim):
        maxx[j] = np.nanmax(x[j])
        minx[j] = np.nanmin(x[j])

    return x, minx, maxx
