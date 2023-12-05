"""
NORMGRAD Finds the normalized gradient field

[DFN,DF,ABSFREG] = NORMGRAD(F,ETA,H) finds the normalized gradient field
of F using ETA as edge parameter and H as voxelsize
Returning the normalized gradients DFN, the derivative DF and the
absolute regularized F, ABSFREG

NB: ETA is the squared ETA

Ex: dfn,df,absfreg = normgrad(f,eta^2,h)
"""

import numpy as np
from .translate_image import translate_image


def normgrad(f, eta, h):
    dim = f.shape
    ndim = len(dim)
    ndim = min(3, ndim)

    # get the values on a staggered grid to avoid central differences
    # [fsgp,fsgn] = staggeredgridvalues(f)

    # gradient
    df = {}
    for i in range(ndim):
        v = np.array([0, 0, 0])
        v[i] = 1
        df[i] = (translate_image(f, v[0], v[1], v[2]) - translate_image(f, -v[0], -v[1], -v[2])) / (2 * h[i])

    absdf2 = np.zeros(dim)

    for i in range(ndim):
        # absdf2 = absdf2 + np.linalg.matrix_power(df[i], 2)
        absdf2 = absdf2 + df[i] ** 2
    absdfreg = np.sqrt(absdf2 + eta ** 2)

    # normalized gradients
    dfn = {}
    for i in range(ndim):
        dfn[i] = np.divide(df[i], absdfreg)

    return dfn, df, absdfreg
