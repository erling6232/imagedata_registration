"""
resize using scipy.ndimage.zoom
"""

from scipy.ndimage import zoom
import numpy as np


class Resize(object):
    """Resize with interpolation to nearest."""

    def __init__(self, f):
        assert isinstance(f, np.ndarray), "Can resize ndarray only (is %s)." % type(f)
        """
        if isinstance(f, tuple):
            self.g = f
            if len(f) == 3:
                # f[slice,rows,columns]
                slice,rows,columns = f
                self.g.shape = (1,slice,rows,columns)
            elif len(f) == 4:
                # f[tag,slice,rows,columns]
                pass
            else:
                raise ValueError("Shape of f has dimension %d." % len(f))
        else:
        """
        self.g = f.copy()  # Dont mess with the original image shape
        if f.ndim == 3:
            # f[slice,rows,columns]
            slice, rows, columns = f.shape
            self.g.shape = (1, slice, rows, columns)
        elif f.ndim == 4:
            # f[tag,slice,rows,columns]
            pass
        else:
            raise ValueError("Shape of f has dimension %d." % f.ndim)
        if __debug__:
            if self.g.ndim != 4:
                raise AssertionError("Copied matrix g should have dim 4 (has %d)." % self.g.ndim)

    # dim = self.g.shape

    def resize(self, dim, method=0):
        self.dim = dim_as_tuple(dim)
        if len(dim) == 3:
            nz, ny, nx = dim
            self.dim = (1, nz, ny, nx)
        elif len(dim) == 4:
            pass
        else:
            raise ValueError("Resize dim must be 3 or 4 (%d given)." % len(dim))

        if isinstance(method, str):
            if method == 'bilinear':
                method = 2
            elif method == 'nearest':
                method = 1
            elif method == 'qubic':
                method = 3

        ratio = np.ones(4)
        for i in range(1, 4):
            # ratio[i] = float(self.g.shape[i]) / dim[i]
            ratio[i] = self.dim[i] / float(self.g.shape[i])

        zoomed_image = zoom(self.g, ratio, order=method)
        if len(dim) == 3:
            # Restore original image 3D shape
            nt, nz, ny, nx = zoomed_image.shape
            zoomed_image.shape = (nz, ny, nx)
        return zoomed_image

    def resizeNearest(self, dim):
        """Resize with nearest interpolation method."""
        return self.resize(dim, method=0)

    def resizeBilinear(self, dim):
        """Resize with bilinear interpolation method."""
        return self.resize(dim, method=1)

    def resizeQubic(self, dim):
        """Resize with qubic interpolation method."""
        return self.resize(dim, method=2)


def dim_as_tuple(dim):
    if isinstance(dim, tuple):
        return dim
    elif isinstance(dim, np.ndarray):
        return tuple(dim)
    else:
        raise ValueError("Converting %s to tuple is not implemented." % type(dim))
