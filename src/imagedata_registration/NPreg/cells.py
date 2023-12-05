"""Cell functions"""

import numpy as np
from .translate_image import translate_image


#
# Computes the innerproduct of two cell arrays as vectors
#
# V = INNERPRODCELL(A,B) computes the innerproduct of two vectors A and B
# as cells. Returns the scalar product in V
#

def innerprodcell(a, b):
    # A and B are matrices
    # print_cell("innerprodcell: a", a)
    # print_cell("innerprodcell: b", b)
    shape_a = cell_shape(a)
    shape_b = cell_shape(b)
    assert shape_a == shape_b, "Cell matrix sizes differ"
    if len(shape_a) == 2 and len(shape_b) == 2:
        # print("innerprodcell: Both A and B are matrices.")
        assert shape_a == shape_b, "Cell matrix sizes differ"
        # matrix inner product as Frobenius norm: <a,b> = trace(b^Ta)
        v = innerprodmatr(a, b)
    elif len(shape_a) == 1 and len(shape_b) == 1:
        # a and b are vectors
        l1 = len(a)
        l2 = len(b)

        # equally long vectors?
        if l1 != l2:
            raise ValueError("Vectors are not equally long (%d/%d)." % (l1, l2))

        # dot product
        v = innerprodvect(a, b, l1)

    elif len(shape_a) == 0 and len(shape_b) == 0:
        pass
    else:
        raise ValueError("Illegal shape of cell matrices.")
    return v


def innerprodmatr(a, b):
    """matrix product: B'*A
    """
    # ab = matrixprodcell(b',a)
    ab = matrixprodcell(ctransposecell(b), a)

    # dim = (len(ab), len(ab[0]))
    dim = cell_shape(ab)
    # print("innerprodmatr: dim", dim)

    # the trace
    v = np.zeros_like(a[0][0])
    for i in range(dim[0]):
        # print("innerprodmatr: ab[%d][%d]" % (i,i), ab[i][i])
        v = v + ab[i][i]

    # print_cell("innerprodmatr: v", v)
    return v


def innerprodvect(a, b, li):
    # print_cell("innerprodvect: a", a)
    v = np.zeros_like(a[0])
    for i in range(li):
        v = v + a[i] * b[i]
    # print_cell("innerprodvect: v", v)
    return v


def hessiancell(f, h):
    """HESSIANCELL Computes the Hessian matrix of an image. Returns the result
    as a cell array

    HESSIANCELL(F,H) computes the hessian matrix of image F and stepsize H.
    Returns the Hessian matrix as a cell array
    """

    dim = f.shape
    ndim = len(dim)

    # print_cell("hessiancell: f", f)
    # pprint.pprint(f)
    # print_cell("hessiancell: h", h)

    # df = cell(ndim,1)
    df = {}

    for i in range(ndim):
        v = np.array([0, 0, 0])
        v[i] = 1
        df[i] = (translate_image(f, v[0], v[1], v[2]) - translate_image(f, -v[0], -v[1], -v[2])) / (2 * h[i])

    # H = cell(ndim,ndim);
    H = {}
    for i in range(ndim):
        H[i] = {}
        for j in range(ndim):
            v = np.array([0, 0, 0])
            v[j] = 1
            H[i][j] = (translate_image(df[i], v[0], v[1], v[2]) -
                       translate_image(df[i], -v[0], -v[1], -v[2])) / (2 * h[j])

    # print_cell("hessiancell: H", H)
    # pprint.pprint(H)
    return H


def multconstcell(A, c):
    """MULTCONSTCELL Multiplies a cell matrix with a constant
    B = MULTCONSTCELL(A,C) multiplies the cell matrix or cell vector A with a
    constant C. Returns the multiplied cell matrix A;
    """
    # print_cell("multconstcell: A", A)
    # print_cell("multconstcell: c", c)
    if type(A) is dict and type(A[0]) is dict:
        m = len(A)
        n = len(A[0])
        B = {}
        for i in range(m):
            B[i] = {}
            for j in range(n):
                B[i][j] = A[i][j] * c
    elif type(A) is dict:
        m = len(A)
        B = {}
        for i in range(m):
            B[i] = A[i] * c
    else:
        B = A * c
    return B


def divideconstcell(A, c):
    """DIVIDECELL(H,A) Divide a cell array by a constant (elementwise)
    """
    if type(A) is dict and type(A[0]) is dict:
        m = len(A)
        n = len(A[0])
        B = {}
        for i in range(m):
            B[i] = {}
            for j in range(n):
                B[i][j] = A[i][j] / c
    elif type(A) is dict:
        m = len(A)
        B = {}
        for i in range(m):
            B[i] = A[i] / c
    else:
        B = A / c
    # print_cell("divideconstcell: B", B)
    return B


def sumcell(a, b):
    """SUMCELL Summing to cell arrays A and B
    C = SUMCELL(A,B) Summing the cell arrays A and B
    """
    assert isinstance(a, type(b)), "Argument type differ"
    assert isinstance(a[0], type(b[0])), "Argument type differ"
    if type(a) is dict and type(a[0]) is dict:
        [rows1, cols1] = (len(a), len(a[0]))
        [rows2, cols2] = (len(b), len(b[0]))
        if rows1 != rows2 or cols1 != cols2:
            raise ValueError("Wrong number of elements")
        rows = rows1
        cols = cols1
        c = {}
        for i in range(rows):
            c[i] = {}
            for j in range(cols):
                c[i][j] = a[i][j] + b[i][j]
    elif type(a) is dict:
        assert type(b) is dict, "Argument b is not a cell structure"
        rows1 = len(a)
        rows2 = len(b)
        if rows1 != rows2:
            raise ValueError("Wrong number of elements")
        rows = rows1
        c = {}
        for i in range(rows):
            c[i] = a[i] + b[i]
    else:
        c = a + b
    # print_cell("sumcell: c", c)
    return c


def matrixprodcell(a, b):
    """MATRIXPRODCELL computes matrixproduct A*B of cell arrays

    AB = MATRIXPRODCELL(A,B) computes matrix product of two cell arrays A and B
    """

    # print_cell("matrixprodcell: a", a)
    # print_cell("matrixprodcell: b", b)
    # ab = a.dot(b)
    # assert type(a) == type(b), "Argument type differ"
    assert isinstance(a, type(b)), "Argument type differ"
    if type(a) is dict and type(a[0]) is dict:
        dima = (len(a), len(a[0]))
        if type(b[0]) is dict:
            dimb = (len(b), len(b[0]))
        else:
            dimb = (len(b), 1)
        dimim = a[0][0].shape

        assert dima[1] == dimb[0], "Dimensions of inner product differ."

        ab = {}
        if type(b[0]) is dict:
            for i in range(dima[0]):
                ab[i] = {}
                for j in range(dimb[1]):
                    ab[i][j] = np.zeros(dimim)
                    for k in range(dimb[0]):
                        ab[i][j] = ab[i][j] + a[i][k] * b[k][j]
        else:
            for i in range(dima[0]):
                ab[i] = {}
                ab[i][0] = np.zeros(dimim)
                for k in range(dimb[0]):
                    ab[i][0] = ab[i][0] + a[i][k] * b[k]
    else:
        ab = a * b
    # assert type(a) is dict, "Argument a is not a cell structure"
    # assert type(b) is dict, "Argument b is not a cell structure"
    # raise ValueError("Matrix product of vector cells not implemented.")
    # print_cell("matrixprodcell: ab", ab)
    return ab


def ctransposecell(a):
    """Conjugate transpose cell array by replacing each element with its conjugate
    """

    # print_cell("ctransposecell: a", a)
    assert type(a) is dict and type(a[0]) is dict, "Argument is not a cell matrix"

    rows, columns = (len(a), len(a[0]))
    b = {}
    for r in range(rows):
        for c in range(columns):
            try:
                b[c][r] = a[r][c]
            except KeyError:
                b[c] = {}
                b[c][r] = a[r][c]
    # print_cell("ctransposecell: b", b)
    return b


def print_cell(str, a):
    if type(a) is dict and type(a[0]) is dict:
        # cell matrix
        m, n = len(a), len(a[0])
        if type(a[0][0]) is np.ndarray:
            print("%s: [%d,%d] cells, %s %s" % (str, m, n, a[0][0].shape, a[0][0].dtype))
        else:
            print("%s: [%d,%d] cells, %s" % (str, m, n, type(a)))
    elif type(a) is dict:
        # cell vector
        m = len(a)
        print("%s: [%d] cells, %s %s" % (str, m, a[0].shape, a[0].dtype))
    else:
        try:
            # ndarray
            print("%s: ndarray, %s %s" % (str, a.shape, a.dtype))
        except AttributeError:
            print("%s:", type(a))


def cell_shape(a):
    if type(a) is dict and type(a[0]) is dict:
        return (len(a), len(a[0]))
    elif type(a) is dict:
        return (len(a),)
    else:
        return ()
