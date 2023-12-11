"""multigrid_nonlin_cc"""

# cython: language_level=3
## cython: language_level=3, boundscheck=False

cimport
cython
cimport
numpy as np
import numpy as np
from .cells import print_cell
from .resize import Resize

DTYPE = np.float64
ctypedef
np.float64_t
DTYPE_t


def multigrid_nonlin(forceu, u_in, prm):
    u = u_in  # Do not modify input

    # prm must contain:
    h = prm['h']
    # cells.print_cell("multigrid_nonlin: h", h)
    dim = prm['dim']
    multigrid = prm['multigrid']
    maxniter = prm['maxniter']
    assert len(multigrid) == len(maxniter), "multigrid and maxniter differ in length"
    level = prm['level']
    assert len(level) == len(maxniter), "level and maxniter differ in length"
    nudim = prm['nudim']
    llambda = prm['lambda']
    mu = prm['mu']
    dt = prm['dt']

    interpmethod = 'bilinear';

    nmultilevel = np.unique(level).size
    dim3 = {}
    for i in range(nmultilevel):
        dim3[i] = dim[i][-3:]

    # initialize u by v
    v = u
    # r = cell(nmultilevel,1)
    # e = cell(nmultilevel,1)
    r = {}
    e = {}

    # nlevel = prm.nlevel
    # multigrid = prm.multigrid
    # maxniter = prm.maxniter
    # level = prm.level
    nlevel = len(level)

    a = range(prm['nudim'])
    noptdim = prm['nudim']
    prmin = {}
    prmin['nudim'] = nudim
    prmin['lambda'] = llambda
    prmin['mu'] = mu
    prmin['dt'] = dt
    for i in range(nlevel):
        l = level[i]

        # make coarser
        if i < nlevel - 1 and multigrid[i] > multigrid[i + 1]:
            ln = level[i + 1]
            prmin['maxniter'] = maxniter[i]

            # solve equation
            # print("multigrid_nonlin: l", l)
            prmin['h'] = h[l]
            u[l] = navlam_nonlinear_cc(forceu[l], v[l], prmin)
            v[l] = u[l]

            # find Av
            av = Au(v[l], h[l], prmin)

            # find residual r
            for j in range(noptdim):
                if l not in r:
                    r[l] = {}
                r[l][a[j]] = forceu[l][a[j]] - av[a[j]]

            # restrict
            for j in range(noptdim):
                # r[ln][a[j]] = resize(r[l][a[j]], dim3[ln], interpmethod)
                rsi = Resize(r[l][a[j]])
                if ln not in r:
                    r[ln] = {}
                r[ln][a[j]] = rsi.resize(dim3[ln], interpmethod)

            for j in range(noptdim):
                # v[ln][a[j]] = resize(v[l][a[j]], dim3[ln], interpmethod)
                rsi = Resize(v[l][a[j]])
                if ln not in v:
                    v[ln] = {}
                v[ln][a[j]] = rsi.resize(dim3[ln], interpmethod)
            continue

        # at the bottom, solve equation
        if multigrid[i] < multigrid[i - 1] and multigrid[i] < multigrid[i + 1]:

            # find Au
            prmin['h'] = h[l]
            av = Au(v[l], h[l], prmin)

            # find new RHS
            for j in range(prm['nudim']):
                forceu[l][a[j]] = av[a[j]] + r[l][a[j]]

            # solve equation
            prmin['h'] = h[l]
            u[l] = navlam_nonlinear_cc(forceu[l], v[l], prmin)

            # find error e
            for j in range(prm['nudim']):
                if l not in e:
                    e[l] = {}
                e[l][a[j]] = u[l][a[j]] - v[l][a[j]]
            continue

        # refine and correct
        if multigrid[i] > multigrid[i - 1] and i > 1:
            lp = level[i - 1]

            # find error e
            for j in range(prm['nudim']):
                # e[l][a[j]] = resize(e[lp][a[j]], dim3[l], interpmethod)
                rsi = Resize(e[lp][a[j]])
                if l not in e:
                    e[l] = {}
                e[l][a[j]] = rsi.resize(dim3[l], interpmethod)

            # correct v by e
            for j in range(prm['nudim']):
                v[l][a[j]] = v[l][a[j]] + e[l][a[j]]

            # relax with initial guess v
            prmin['h'] = h[l]
            u[l] = navlam_nonlinear_cc(forceu[l], v[l], prmin)
    return u[0]


# ----------------------------------------------------------------

def Au(u, h, prm):
    # prm needs to contain
    mu = prm['mu']
    llambda = prm['lambda']
    nudim = prm['nudim']

    H = np.zeros((nudim, nudim))
    for j in range(nudim):
        for k in range(nudim):
            H[j, k] = h[j] * h[k]

    # F = cell(nudim,1);
    F = {}
    if prm['nudim'] == 2:

        assert len(u[0].shape) >= 2, "Shape of u[0] is not 2+ dim"
        assert len(u[1].shape) >= 2, "Shape of u[1] is not 2+ dim"
        u0_shape = u[0].shape
        u1_shape = u[1].shape
        if len(u0_shape) == 2:
            u[0].shape = (1, u0_shape[0], u0_shape[1])
        if len(u1_shape) == 2:
            u[1].shape = (1, u1_shape[0], u1_shape[1])

        assert u[0].shape == u[1].shape, "Shape of u[0] and u[1] differ."

        nz, ny, nx = u[0].shape

        d0 = u[0] * (-2 * mu / H[1, 1] - 2 * (llambda + 2 * mu) / H[0, 0])
        dy1 = u[0][:, np.r_[1:ny, -1], :] * ((llambda + 2 * mu) / H[0, 0])
        dy2 = u[0][:, np.r_[0, :ny - 1], :] * ((llambda + 2 * mu) / H[0, 0])
        dx1 = u[0][:, :, np.r_[1:nx, -1]] * (mu / H[1, 1])
        dx2 = u[0][:, :, np.r_[0, :nx - 1]] * (mu / H[1, 1])
        temp = u[1][:, :, np.r_[1:nx - 1]]
        dyx1 = temp[:, np.r_[1:ny, -1], :] * ((llambda + mu) / (4 * H[0, 1]))
        temp = u[1][:, :, np.r_[0, :nx - 1]]
        dyx2 = temp[:, np.r_[1:ny, -1], :] * (-(llambda + mu) / (4 * H[0, 1]))
        temp = u[1][:, :, np.r_[1:nx, -1]]
        dyx3 = temp[:, np.r_[0, :ny - 1], :] * (-(llambda + mu) / (4 * H[0, 1]))
        temp = u[1][:, :, np.r_[0, :nx - 1]]
        dyx4 = temp[:, np.r_[0, :ny - 1], :] * ((llambda + mu) / (4 * H[0, 1]))
        F[0] = d0 + dy1 + dy2 + dx1 + dx2 + dyx1 + dyx2 + dyx3 + dyx4

        d0 = u[1] * (-2 * mu / H[1, 1] - 2 * (llambda + 2 * mu) / H[2, 2])
        dy1 = u[1][:, np.r_[1:ny, -1], :] * (mu / H[1, 1])
        dy2 = u[1][:, np.r_[0, :ny - 1], :] * (mu / H[1, 1])
        dx1 = u[1][:, :, np.r_[1:nx, -1]] * ((llambda + 2 * mu) / H[2, 2])
        dx2 = u[1][:, :, np.r_[0, :nx - 1]] * ((llambda + 2 * mu) / H[2, 2])
        temp = u[0][:, :, np.r_[1:nx, -1]]
        dyx1 = temp[:, np.r_[1:ny, -1], :] * ((llambda + mu) / (4 * H[1, 2]))
        temp = u[0][:, :, np.r_[0, :nx - 1]]
        dyx2 = temp[:, np.r_[1:ny, -1], :] * (-(llambda + mu) / (4 * H[1, 2]))
        temp = u[0][:, :, np.r_[1:nx, -1]]
        dyx3 = temp[:, np.r_[0, :ny - 1], :] * (-(llambda + mu) / (4 * H[1, 2]))
        temp = u[0][:, :, np.r_[0, :nx - 1]]
        dyx4 = temp[:, np.r_[0, :ny - 1], :] * ((llambda + mu) / (4 * H[1, 2]))
        F[1] = d0 + dy1 + dy2 + dx1 + dx2 + dyx1 + dyx2 + dyx3 + dyx4

        u[0].shape = u0_shape
        u[1].shape = u1_shape

    elif prm['nudim'] == 3:

        assert len(u[0].shape) >= 3, "Shape of u[0] is not 3+ dim"
        assert len(u[1].shape) >= 3, "Shape of u[1] is not 3+ dim"
        assert len(u[2].shape) >= 3, "Shape of u[2] is not 3+ dim"
        u0_shape = u[0].shape
        u1_shape = u[1].shape
        u2_shape = u[2].shape
        if len(u0_shape) == 3:
            u[0].shape = (1, u0_shape[0], u0_shape[1], u0_shape[2])
        if len(u1_shape) == 3:
            u[1].shape = (1, u1_shape[0], u1_shape[1], u1_shape[2])
        if len(u2_shape) == 3:
            u[2].shape = (1, u2_shape[0], u2_shape[1], u2_shape[2])

        assert u[0].shape == u[1].shape, "Shape of u[0] and u[1] differ."
        assert u[0].shape == u[2].shape, "Shape of u[0] and u[2] differ."

        for i in range(3):
            assert u[i].shape[-3:] == forceu[i].shape[-3:],(
                "Shape of u[{}] and forceu[{}] differ.".format(i, i))

        nt, nz, ny, nx = u[0].shape

        d0 = u[0] * (-2 * mu * (1 / H[0, 0] + 1 / H[1, 1] + 1 / H[2, 2]) - 2 * (mu + llambda) / H[0, 0])
        dz1 = u[0][:, np.r_[1:nz, -1], :, :] * (mu / H[0, 0] + (llambda + mu) / H[0, 0])
        dz2 = u[0][:, np.r_[0, :nz - 1], :, :] * (mu / H[0, 0] + (llambda + mu) / H[0, 0])
        dy1 = u[0][:, :, np.r_[1:ny, -1], :] * (mu / H[1, 1])
        dy2 = u[0][:, :, np.r_[0, :ny - 1], :] * (mu / H[1, 1])
        dx1 = u[0][:, :, :, np.r_[1:nx, -1]] * (mu / H[2, 2])
        dx2 = u[0][:, :, :, np.r_[0, :nx - 1]] * (mu / H[2, 2])
        temp = u[1][:, :, np.r_[1:ny, -1], :]
        dzy1 = temp[:, np.r_[1:nz, -1], :, :] * ((llambda + mu) / (4 * H[0, 1]))
        temp = u[1][:, :, np.r_[0, :ny - 1], :]
        dzy2 = temp[:, np.r_[1:nz, -1], :, :] * (-(llambda + mu) / (4 * H[0, 1]))
        temp = u[1][:, :, np.r_[1:ny, -1], :]
        dzy3 = temp[:, np.r_[0, :nz - 1], :, :] * (-(llambda + mu) / (4 * H[0, 1]))
        temp = u[1][:, :, np.r_[0, :ny - 1], :]
        dzy4 = temp[:, np.r_[0, :nz - 1], :, :] * ((llambda + mu) / (4 * H[0, 1]))
        temp = u[2][:, :, :, np.r_[1:nx, -1]]
        dzx1 = temp[:, np.r_[1:nz, -1], :, :] * ((llambda + mu) / (4 * H[0, 2]))
        temp = u[2][:, :, :, np.r_[0, :nx - 1]]
        dzx2 = temp[:, np.r_[1:nz, -1], :, :] * (-(llambda + mu) / (4 * H[0, 2]))
        temp = u[2][:, :, :, np.r_[1:nx, -1]]
        dzx3 = temp[:, np.r_[0, :nz - 1], :, :] * (-(llambda + mu) / (4 * H[0, 2]))
        temp = u[2][:, :, :, np.r_[0, :nx - 1]]
        dzx4 = temp[:, np.r_[0, :nz - 1], :, :] * ((llambda + mu) / (4 * H[0, 2]))
        F[0] = d0 + dz1 + dz2 + dy1 + dy2 + dx1 + dx2 + dzy1 + dzy2 + dzy3 + dzy4 + dzx1 + dzx2 + dzx3 + dzx4

        d0 = u[1] * (-2 * mu * (1 / H[0, 0] + 1 / H[1, 1] + 1 / H[2, 2]) - 2 * (mu + llambda) / H[1, 1])
        dz1 = u[1][:, np.r_[1:nz, -1], :, :] * (mu / H[0, 0])
        dz2 = u[1][:, np.r_[0, :nz - 1], :, :] * (mu / H[0, 0])
        dy1 = u[1][:, :, np.r_[1:ny, -1], :] * (mu / H[1, 1] + (llambda + mu) / H[1, 1])
        dy2 = u[1][:, :, np.r_[0, :ny - 1], :] * (mu / H[1, 1] + (llambda + mu) / H[1, 1])
        dx1 = u[1][:, :, :, np.r_[1:nx, -1]] * (mu / H[2, 2])
        dx2 = u[1][:, :, :, np.r_[0, :nx - 1]] * (mu / H[2, 2])
        temp = u[0][:, :, np.r_[1:ny, -1], :]
        dzy1 = temp[:, np.r_[1:nz, -1], :, :] * ((llambda + mu) / (4 * H[0, 1]))
        temp = u[0][:, :, np.r_[0, :ny - 1], :]
        dzy2 = temp[:, np.r_[1:nz, -1], :, :] * (-(llambda + mu) / (4 * H[0, 1]))
        temp = u[0][:, :, np.r_[1:ny, -1], :]
        dzy3 = temp[:, np.r_[0, :nz - 1], :, :] * (-(llambda + mu) / (4 * H[0, 1]))
        temp = u[0][:, :, np.r_[0, :ny - 1], :]
        dzy4 = temp[:, np.r_[0, :nz - 1], :, :] * ((llambda + mu) / (4 * H[0, 1]))
        temp = u[2][:, :, :, np.r_[1:nx, -1]]
        dyx1 = temp[:, :, np.r_[1:ny, -1], :] * ((llambda + mu) / (4 * H[1, 2]))
        temp = u[2][:, :, :, np.r_[0, :nx - 1]]
        dyx2 = temp[:, :, np.r_[1:ny, -1], :] * (-(llambda + mu) / (4 * H[1, 2]))
        temp = u[2][:, :, :, np.r_[1:nx, -1]]
        dyx3 = temp[:, :, np.r_[0, :ny - 1], :] * (-(llambda + mu) / (4 * H[1, 2]))
        temp = u[2][:, :, :, np.r_[0, :nx - 1]]
        dyx4 = temp[:, :, np.r_[0, :ny - 1], :] * ((llambda + mu) / (4 * H[1, 2]))
        F[1] = d0 + dz1 + dz2 + dy1 + dy2 + dx1 + dx2 + dzy1 + dzy2 + dzy3 + dzy4 + dyx1 + dyx2 + dyx3 + dyx4

        d0 = u[2] * (-2 * mu * (1 / H[0, 0] + 1 / H[1, 1] + 1 / H[2, 2]) - 2 * (mu + llambda) / H[2, 2])
        dz1 = u[2][:, np.r_[1:nz, -1], :, :] * (mu / H[0, 0])
        dz2 = u[2][:, np.r_[0, :nz - 1], :, :] * (mu / H[0, 0])
        dy1 = u[2][:, :, np.r_[1:ny, -1], :] * (mu / H[1, 1])
        dy2 = u[2][:, :, np.r_[0, :ny - 1], :] * (mu / H[1, 1])
        dx1 = u[2][:, :, :, np.r_[1:nx, -1]] * (mu / H[2, 2] + (llambda + mu) / H[2, 2])
        dx2 = u[2][:, :, :, np.r_[0, :nx - 1]] * (mu / H[2, 2] + (llambda + mu) / H[2, 2])
        dzx1 = u[0][:, np.r_[1:nz, -1], :, np.r_[1:nx, -1]] * ((llambda + mu) / (4 * H[0, 2]))
        dzx2 = u[0][:, np.r_[1:nz, -1], :, np.r_[0, :nx - 1]] * (-(llambda + mu) / (4 * H[0, 2]))
        dzx3 = u[0][:, np.r_[0, :nz - 1], :, np.r_[1:nx, -1]] * (-(llambda + mu) / (4 * H[0, 2]))
        dzx4 = u[0][:, np.r_[0, :nz - 1], :, np.r_[0, :nx - 1]] * ((llambda + mu) / (4 * H[0, 2]))
        dyx1 = u[1][:, :, np.r_[1:ny, -1], np.r_[1:nx, -1]] * ((llambda + mu) / (4 * H[1, 2]))
        dyx2 = u[1][:, :, np.r_[1:ny, -1], np.r_[0, :nx - 1]] * (-(llambda + mu) / (4 * H[1, 2]))
        dyx3 = u[1][:, :, np.r_[0, :ny - 1], np.r_[1:nx, -1]] * (-(llambda + mu) / (4 * H[1, 2]))
        dyx4 = u[1][:, :, np.r_[0, :ny - 1], np.r_[0, :nx - 1]] * ((llambda + mu) / (4 * H[1, 2]))
        F[2] = d0 + dz1 + dz2 + dy1 + dy2 + dx1 + dx2 + dzx1 + dzx2 + dzx3 + dzx4 + dyx1 + dyx2 + dyx3 + dyx4

        u[0].shape = u0_shape
        u[1].shape = u1_shape
        u[2].shape = u2_shape

    else:
        raise ValueError("prm.nudim out of range: %d" % prm['nudim'])

    # # stabilizing factor
    # for i = 1 : prm.nudim
    #     F{i} = prm.dt*F{i} + u{i};
    # end;

    return F


# ----------------------------------------------------------------

# cdefine the signature of our c function
cdef
extern
from

"navlam_nonlinear_cc.h":
void
navlam_nonlinear_cc3(double * forceu0, double * forceu1, double * forceu2,
                     double * u0, double * u1, double * u2,
                     int
maxniter, double * h, int
nudim, double
llambda, double
mu, double
dt,
size_t
nz, size_t
ny, size_t
nx)

def navlam_nonlinear_cc(forceu, u, prm):
    if prm['nudim'] == 2:
        return navlam_nonlinear2(forceu, u, prm)
    elif prm['nudim'] == 3:
        assert forceu[0].flags['C_CONTIGUOUS']
        assert forceu[1].flags['C_CONTIGUOUS']
        assert forceu[2].flags['C_CONTIGUOUS']
        assert u[0].flags['C_CONTIGUOUS']
        assert u[1].flags['C_CONTIGUOUS']
        assert u[2].flags['C_CONTIGUOUS']
        # forceu0 = np.ascontiguousarray(forceu[0])
        # forceu1 = np.ascontiguousarray(forceu[1])
        # forceu2 = np.ascontiguousarray(forceu[2])
        # u0 = np.ascontiguousarray(u[0])
        # u1 = np.ascontiguousarray(u[1])
        # u2 = np.ascontiguousarray(u[2])
        # u[0][3,5,7] = 357
        # u[0][4,2,6] = 426
        nz, ny, nx = u[0].shape
        navlam_nonlinear_cc3(
        < double * > np.PyArray_DATA(forceu[0]),
        < double * > np.PyArray_DATA(forceu[1]),
        < double * > np.PyArray_DATA(forceu[2]),
        < double * > np.PyArray_DATA(u[0]),
        < double * > np.PyArray_DATA(u[1]),
        < double * > np.PyArray_DATA(u[2]),
          prm['maxniter'][0],
        < double * > np.PyArray_DATA(prm['h']),
          prm['nudim'],
          float(prm['lambda']),
          float(prm['mu']),
          float(prm['dt']),
          nz, ny, nx
        )
        # return {0: u0, 1: u1, 2: u2}
        return u


def navlam_nonlinear2(forceu, u_in, prm):
    """
    Fix point iterations (isolating the unknown on left hand side and
    iterating). See page 100 in 'A multigrid tutorial'
    """
    """
    for i = 1 : maxniter
        F{1} = u{1}([2:end end],:,:)*((lambda+2*mu)/H(1,1)) + ...
            u{1}([1 1:end-1],:,:)*((lambda+2*mu)/H(1,1)) + ...
            u{1}(:,[2:end end],:)*(mu/H(2,2)) + ...
            u{1}(:,[1 1:end-1],:)*(mu/H(2,2)) + ...
            u{2}([2:end end],[2:end end],:)*((lambda+mu)/(4*H(1,2))) + ...
            u{2}([2:end end],[1 1:end-1],:)*(-(lambda+mu)/(4*H(1,2))) + ...
            u{2}([1 1:end-1],[2:end end],:)*(-(lambda+mu)/(4*H(1,2))) + ...
            u{2}([1 1:end-1],[1 1:end-1],:)*((lambda+mu)/(4*H(1,2))) - ...
            forceu{1};
        
        % put on right hand side and divide by the term in front of u_ijk
        F{1} = -F{1}/(-2*mu*(1/H(1,1) + 1/H(2,2))-2*(lambda+mu)/H(1,1));
        
        F{2} =  u{2}([2:end end],:,:)*(mu/H(1,1)) + ...
            u{2}([1 1:end-1],:,:)*(mu/H(1,1)) + ...
            u{2}(:,[2:end end],:)*((lambda+2*mu)/H(2,2)) + ...
            u{2}(:,[1 1:end-1],:)*((lambda+2*mu)/H(2,2)) + ...
            u{1}([2:end end],[2:end end],:)*((lambda+mu)/(4*H(1,2))) + ...
            u{1}([2:end end],[1 1:end-1],:)*(-(lambda+mu)/(4*H(1,2))) + ...
            u{1}([1 1:end-1],[2:end end],:)*(-(lambda+mu)/(4*H(1,2))) + ...
            u{1}([1 1:end-1],[1 1:end-1],:)*((lambda+mu)/(4*H(1,2))) - ...
            forceu{2};           
        
        % put on right hand side and divide by the term in front of u_ijk
        F{2} = -F{2}/(-2*mu*(1/H(1,1) + 1/H(2,2))-2*(lambda+mu)/H(2,2));
        
        % pix point iterations
        u{1} = F{1};
        u{2} = F{2};        
        
    end;
    """

    assert u_in[0].dtype == DTYPE
    assert u_in[1].dtype == DTYPE
    assert u_in[2].dtype == DTYPE

    u = u_in.copy()  # Do not modify input

    # Py_ssize_t is the proper C type for Python array indices.
    cdef
    Py_ssize_t
    ny, nx
    cdef
    int
    maxniter, i
    cdef
    size_t
    j, k

    # prm must contain
    if type(prm['maxniter']) is tuple:
        maxniter = prm['maxniter'][0]
    else:
        maxniter = prm['maxniter']
    h = prm['h']
    cdef
    int
    llambda = prm['lambda']
    cdef
    int
    mu = prm['mu']
    cdef
    int
    dt = prm['dt']
    cdef
    int
    nudim = prm['nudim']

    # F = cell(prm['nudim'],1)
    F = {}
    # cdef H ndarray float64
    # cdef DTYPE_t[:, :] H = np.zeros((prm['nudim'],prm['nudim']))
    # cdef DTYPE_t [:, :] H = np.zeros([prm['nudim'],prm['nudim']], dtype=DTYPE)
    assert prm['nudim'] == 2
    cdef
    DTYPE_t
    Harr[2][2]
    cdef
    DTYPE_t[:, :]
    H = Harr
    for j in range(prm['nudim']):
        for k in range(prm['nudim']):
            H[j, k] = h[j] * h[k]

    assert len(u[0].shape) >= 2, "Shape of u[0] is not 2+ dim"
    assert len(u[1].shape) >= 2, "Shape of u[1] is not 2+ dim"
    u0_shape = u[0].shape
    u1_shape = u[1].shape
    if len(u0_shape) == 2:
        u[0].shape = (1, u0_shape[0], u0_shape[1])
    if len(u1_shape) == 2:
        u[1].shape = (1, u1_shape[0], u1_shape[1])

    assert u[0].shape == u[1].shape, "Shape of u[0] and u[1] differ."

    ny, nx = u[0].shape

    for i in range(maxniter):
        dy1 = u[0][:, np.r_[1:ny, -1], :] * ((llambda + 2 * mu) / H[0, 0])
        dy2 = u[0][:, np.r_[0, :ny - 1], :] * ((llambda + 2 * mu) / H[0, 0])
        dx1 = u[0][:, :, np.r_[1:nx, -1]] * (mu / H[1, 1])
        dx2 = u[0][:, :, np.r_[0, :nx - 1]] * (mu / H[1, 1])
        temp = u[1][:, :, np.r_[1:nx, -1]]
        dyx1 = temp[:, np.r_[1:ny, -1], :] * ((llambda + mu) / (4 * H[0, 1]))
        temp = u[1][:, :, np.r_[0, :nx - 1]]
        dyx2 = temp[:, np.r_[1:ny, -1], :] * (-(llambda + mu) / (4 * H[0, 1]))
        temp = u[1][:, :, np.r_[1:nx, -1]]
        dyx3 = temp[:, np.r_[0, :ny - 1], :] * (-(llambda + mu) / (4 * H[0, 1]))
        temp = u[1][:, :, np.r_[0, :nx - 1]]
        dyx4 = temp[:, np.r_[0, :ny - 1], :] * ((llambda + mu) / (4 * H[0, 1]))
        F[0] = dy1 + dy2 + dx1 + dx2 + dyx1 + dyx2 + dyx3 + dyx4 - forceu[0]

        # put on right hand side and divide by the term in front of u_ijk
        F[0] = -F[0] / (-2 * mu * (1 / H[0, 0] + 1 / H[1, 1]) - 2 * (llambda + mu) / H[0, 0])

        dy1 = u[1][:, np.r_[1:ny, -1], :] * (mu / H[0, 0])
        dy2 = u[1][:, np.r_[0, :ny - 1], :] * (mu / H[0, 0])
        dx1 = u[1][:, :, np.r_[1:nx, -1]] * ((llambda + 2 * mu) / H[1, 1])
        dx2 = u[1][:, :, np.r_[0, :nx - 1]] * ((llambda + 2 * mu) / H[1, 1])
        temp = u[0][:, :, np.r_[1:nx, -1]]
        dyx1 = temp[:, np.r_[1:ny, -1], :] * ((llambda + mu) / (4 * H[0, 1]))
        temp = u[0][:, :, np.r_[0, :nx - 1]]
        dyx2 = temp[:, np.r_[1:ny, -1], :] * (-(llambda + mu) / (4 * H[0, 1]))
        temp = u[0][:, :, np.r_[1:nx, -1]]
        dyx3 = temp[:, np.r_[0, :ny - 1], :] * (-(llambda + mu) / (4 * H[0, 1]))
        temp = u[0][:, :, np.r_[0, :nx - 1]]
        dyx4 = temp[:, np.r_[0, :ny - 1], :] * ((llambda + mu) / (4 * H[0, 1]))
        F[1] = dy1 + dy2 + dx1 + dx2 + dyx1 + dyx2 + dyx3 + dyx4 - forceu[1]

        # put on right hand side and divide by the term in front of u_ijk
        F[1] = -F[1] / (-2 * mu * (1 / H[0, 0] + 1 / H[1, 1]) - 2 * (llambda + mu) / H[1, 1])

        # pix point iterations
        u[0] = F[0]
        u[1] = F[1]

    u[0].shape = u0_shape
    u[1].shape = u1_shape

    return u
