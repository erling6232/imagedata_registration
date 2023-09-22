"""
% TRANSIM Translate image for use in derivatives
%
%   U = TRANSIM(U,DI,DJ,DK) Translates image for use in derivatives, returns
%   the translated image U. DI, DJ, DK are integer valued translations.
"""

import numpy as np
import math


def sign(x):
    return math.copysign(1, x)
    # return cmp(x, 0)


def translate_image(u, dslice, drow, dcolumn, dtag=None):
    # [M N O P] = u.shape	# [tag,slice,rows,columns]
    if len(u.shape) == 4:
        [tags, slices, rows, columns] = u.shape
    elif len(u.shape) == 3:
        [slices, rows, columns] = u.shape
        tags = 1
    else:
        raise ValueError("Dimension of matrix must be 3 or 4 (is %d)." % len(u.shape))

    di = int(sign(drow) * min(rows, abs(drow)))
    dj = int(sign(dcolumn) * min(columns, abs(dcolumn)))
    dk = int(sign(dslice) * min(slices, abs(dslice)))
    if dtag is not None:
        if len(u.shape) != 4:
            raise ValueError("Dimension of matrix must be 4 (is %d) when dtag is specified."
                             % len(u.shape))
        dl = sign(dtag) * min(tags, abs(dtag))

    if di > 0:
        indrows = np.concatenate((np.arange(di, rows),
                                  (rows - 1) * np.ones(di, dtype=np.intp)))
    elif di == 0:
        indrows = np.arange(rows)
    elif di < 0:
        indrows = np.concatenate((np.zeros(-di, dtype=np.intp), np.arange(rows + di)))

    if dj > 0:
        indcolumns = np.concatenate((np.arange(dj, columns),
                                     (columns - 1) * np.ones(dj, dtype=np.intp)))
    elif dj == 0:
        indcolumns = np.arange(columns)
    elif dj < 0:
        indcolumns = np.concatenate((np.zeros(-dj, dtype=np.intp), np.arange(columns + dj)))

    if dk > 0:
        indslices = np.concatenate((np.arange(dk, slices),
                                    (slices - 1) * np.ones(dk, dtype=np.intp)))
    elif dk == 0:
        indslices = np.arange(slices)
    elif dk < 0:
        indslices = np.concatenate((np.zeros(-dk, dtype=np.intp), np.arange(slices + dk)))

    # the image to return
    if dtag is not None:
        if dl > 0:
            indtags = np.concatenate((np.arange(dl, tags),
                                      (tags - 1) * np.ones(dl, dtype=np.intp)))
        elif dl == 0:
            indtags = np.arange(tags)
        elif dl < 0:
            indtags = np.concatenate((np.zeros(-dl, dtype=np.intp), np.arange(tags + dl)))
        return u[np.ix_(indtags, indslices, indrows, indcolumns)]
    else:
        if len(u.shape) == 4:
            indtags = np.arange(u.shape[0])
            return u[np.ix_(indtags, indslices, indrows, indcolumns)]
        else:
            return u[np.ix_(indslices, indrows, indcolumns)]
