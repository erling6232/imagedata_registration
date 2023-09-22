"""
transform
APPLYTRANSFORM Applies a transformation made by FLUIDREG

FU = APPLYTRANSFORM(F,U,X,EXT,INTERPMETHOD) applied the transform in U
to image F. F is a scalar image and U is a cell array of transformations
in X,Y,Z (matrix notation, as in NDGRID). X specifies the grid
(normally made by np.meshgrid). EXT specifies the value to use outside the
image domain from interpolation. INTERPMETHOD specifies the interpolation
method.
"""

# import numpy as np
# from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.interpolate import interpn
# import pprint


class Transform(object):
    """Transform with interpolation to nearest."""

    def __init__(self, ext):
        self.ext = ext


def print_grid(str, f):
    g = f.copy()
    i = 0
    slices, rows, columns = g.shape
    g.shape = g.size
    for slice in range(slices):
        for row in range(rows):
            for column in range(columns):
                print("%s: %d: (%d,%d,%d) %f" % (str, i, slice, row, column, g[i]))
                i += 1
    g.shape = (slices, rows, columns)


class TransformLinear(Transform):
    def apply(self, f, udim, u, x):
        # dim = size(f)
        # print("x[]"); pprint.pprint(x)
        # print("u[]"); pprint.pprint(u)
        ndim = udim
        xi = {}
        n1 = x['size']
        n2 = len(u)
        # print("TransformLinear.apply: len(x)", n1, ", len(u)", n2)
        # print("TransformLinear.apply: f.shape", f.shape)
        n = min(n1, n2)
        # for i in u.keys():
        for i in range(n):
            xi[i] = x[i] + u[i]

        # print("xi[]"); pprint.pprint(xi)

        # for i in u.keys():
        for i in range(n):
            # xi[i](xi[i] < self.ext.minx(i)+0.01) = self.ext.minx(i)+0.01
            # xi[i](xi[i] > self.ext.maxx(i)-0.01) = self.ext.maxx(i)-0.01
            xi[i][xi[i] < self.ext.minx[i]] = self.ext.minx[i] + 0.01
            xi[i][xi[i] > self.ext.maxx[i]] = self.ext.maxx[i] - 0.01

        # transform the image, f evaluated at x + u
        if ndim == 2:
            g = interpn((x['row'], x['column']),
                        f,
                        (xi[0], xi[1]),
                        method='linear',
                        bounds_error=False)
        elif ndim == 3:
            # print_grid("f", f)
            g = interpn((x['slice'], x['row'], x['column']),
                        f,
                        (xi[0], xi[1], xi[2]),
                        method='linear',
                        bounds_error=False)
        # print_grid("g", g)
        elif ndim == 4:
            g = interpn((x['tag'], x['slice'], x['row'], x['column']),
                        f,
                        (xi[0], xi[1], xi[2], xi[3]),
                        method='linear',
                        bounds_error=False)
        else:
            raise ValueError("Wrong ndim %d." % ndim)
        return g

    """ method='nearest', method='cubic' """
