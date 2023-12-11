"""multigrid_nonlin_cy"""

# cython: language_level=3
## cython: language_level=3, boundscheck=False

cimport cython
cimport numpy as np
import numpy as np
import pyximport
from .cells import print_cell
from .resize import Resize

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# pyximport.install(setup_args={"script_args":["--compiler=mingw32"],
pyximport.install(setup_args={
    "include_dirs": np.get_include()},
    reload_support=True)


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
        prmin['maxniter'] = maxniter[i]

        # make coarser
        if i < nlevel - 1 and multigrid[i] > multigrid[i + 1]:
            ln = level[i + 1]

            # solve equation
            # print("multigrid_nonlin: l", l)
            prmin['h'] = h[l]
            if prm['nudim'] == 2:
                u[l] = navlam_nonlinear2(forceu[l], v[l], prmin)
            elif prm['nudim'] == 3:
                u[l] = navlam_nonlinear3(forceu[l], v[l], prmin)
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
            if prm['nudim'] == 2:
                u[l] = navlam_nonlinear2(forceu[l], v[l], prmin)
            elif prm['nudim'] == 3:
                u[l] = navlam_nonlinear3(forceu[l], v[l], prmin)

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
            if prm['nudim'] == 2:
                u[l] = navlam_nonlinear2(forceu[l], v[l], prmin)
            elif prm['nudim'] == 3:
                u[l] = navlam_nonlinear3(forceu[l], v[l], prmin)
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
        # dzx1 = u[0][:, np.r_[1:nz, -1], :, np.r_[1:nx, -1]] * ((llambda + mu) / (4 * H[0, 2]))
        temp = u[0][:, :, :, np.r_[1:nx, -1]]
        dzx1 = temp[:, np.r_[1:nz, -1], :, :] * ((llambda + mu) / (4 * H[0, 2]))
        # dzx2 = u[0][:, np.r_[1:nz, -1], :, np.r_[0, :nx - 1]] * (-(llambda + mu) / (4 * H[0, 2]))
        temp = u[0][:, :, :, np.r_[0, :nx - 1]]
        dzx2 = temp[:, np.r_[1:nz, -1], :, :] * (-(llambda + mu) / (4 * H[0, 2]))
        # dzx3 = u[0][:, np.r_[0, :nz - 1], :, np.r_[1:nx, -1]] * (-(llambda + mu) / (4 * H[0, 2]))
        temp = u[0][:, :, :, np.r_[1:nx, -1]]
        dzx3 = temp[:, np.r_[0, :nz - 1], :, :] * (-(llambda + mu) / (4 * H[0, 2]))
        # dzx4 = u[0][:, np.r_[0, :nz - 1], :, np.r_[0, :nx - 1]] * ((llambda + mu) / (4 * H[0, 2]))
        temp = u[0][:, :, :, np.r_[0, :nx - 1]]
        dzx4 = temp[:, np.r_[0, :nz - 1], :, :] * ((llambda + mu) / (4 * H[0, 2]))
        # dyx1 = u[1][:, :, np.r_[1:ny, -1], np.r_[1:nx, -1]] * ((llambda + mu) / (4 * H[1, 2]))
        temp = u[1][:, :, :, np.r_[1:nx, -1]]
        dyx1 = temp[:, :, np.r_[1:ny, -1], :] * ((llambda + mu) / (4 * H[1, 2]))
        # dyx2 = u[1][:, :, np.r_[1:ny, -1], np.r_[0, :nx - 1]] * (-(llambda + mu) / (4 * H[1, 2]))
        temp = u[1][:, :, :, np.r_[0, :nx - 1]]
        dyx2 = temp[:, :, np.r_[1:ny, -1], :] * (-(llambda + mu) / (4 * H[1, 2]))
        # dyx3 = u[1][:, :, np.r_[0, :ny - 1], np.r_[1:nx, -1]] * (-(llambda + mu) / (4 * H[1, 2]))
        temp = u[1][:, :, :, np.r_[1:nx, -1]]
        dyx3 = temp[:, :, np.r_[0, :ny - 1], :] * (-(llambda + mu) / (4 * H[1, 2]))
        # dyx4 = u[1][:, :, np.r_[0, :ny - 1], np.r_[0, :nx - 1]] * ((llambda + mu) / (4 * H[1, 2]))
        temp = u[1][:, :, :, np.r_[0, :nx - 1]]
        dyx4 = temp[:, :, np.r_[0, :ny - 1], :] * ((llambda + mu) / (4 * H[1, 2]))
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

"""
def navlam_nonlinear(forceu, u_in, prm):

    # THIS IMPLEMENTATION DOES NOT WORK IN PYTHON

    # Fix point iterations (isolating the unknown on left hand side and
    # iterating). See page 100 in 'A multigrid tutorial'


    u = u_in.copy() # Do not modify input

    # prm must contain
    if type(prm['maxniter']) is tuple:
        maxniter = prm['maxniter'][0]
    else:
        maxniter = prm['maxniter']
    h = prm['h']
    llambda = prm['lambda']
    mu = prm['mu']
    dt = prm['dt']

    # F = cell(prm['nudim'],1)
    F = {}
    H = np.zeros((prm['nudim'],prm['nudim']))
    for j in range(prm['nudim']):
        for k in range(prm['nudim']):
            H[j,k] = h[j]*h[k]

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

        nz,ny,nx=u[0].shape

        for i in range(maxniter):

            F[0] = u[0][:,np.r_[1:ny,-1],:]*((llambda+2*mu)/H[0,0]) + \
            u[0][:,np.r_[0,:ny-1],:]*((llambda+2*mu)/H[0,0]) + \
            u[0][:,:,np.r_[1:nx,-1]]*(mu/H[1,1]) + \
            u[0][:,:,np.r_[0,:nx-1]]*(mu/H[1,1]) + \
            u[1][:,np.r_[1:ny,-1],np.r_[1:nx,-1]]*((llambda+mu)/(4*H[0,1])) + \
            u[1][:,np.r_[1:ny,-1],np.r_[0,:nx-1]]*(-(llambda+mu)/(4*H[0,1])) + \
            u[1][:,np.r_[0,:ny-1],np.r_[1:nx,-1]]*(-(llambda+mu)/(4*H[0,1])) + \
            u[1][:,np.r_[0,:ny-1],np.r_[0,:nx-1]]*((llambda+mu)/(4*H[0,1])) - \
            forceu[0]
        
            # put on right hand side and divide by the term in front of u_ijk
            F[0] = -F[0]/(-2*mu*(1/H[0,0] + 1/H[1,1])-2*(llambda+mu)/H[0,0])
        
            F[1] =  u[1][:,np.r_[1:ny,-1],:]*(mu/H[0,0]) + \
            u[1][:,np.r_[0,:ny-1],:]*(mu/H[0,0]) + \
            u[1][:,:,np.r_[1:nx,-1]]*((llambda+2*mu)/H[1,1]) + \
            u[1][:,:,np.r_[0,:nx-1]]*((llambda+2*mu)/H[1,1]) + \
            u[0][:,np.r_[1:ny,-1],np.r_[1:nx,-1]]*((llambda+mu)/(4*H[0,1])) + \
            u[0][:,np.r_[1:ny,-1],np.r_[0,:nx-1]]*(-(llambda+mu)/(4*H[0,1])) + \
            u[0][:,np.r_[0,:ny-1],np.r_[1:nx,-1]]*(-(llambda+mu)/(4*H[0,1])) + \
            u[0][:,np.r_[0,:ny-1],np.r_[0,:nx-1]]*((llambda+mu)/(4*H[0,1])) - \
            forceu[1]
        
            # put on right hand side and divide by the term in front of u_ijk
            F[1] = -F[1]/(-2*mu*(1/H[0,0] + 1/H[1,1])-2*(llambda+mu)/H[1,1])

            # pix point iterations
            u[0] = F[0]        
            u[1] = F[1]

        u[0].shape = u0_shape
        u[1].shape = u1_shape
        
    elif prm['nudim'] == 3:

        #cells.print_cell("navlam_nonlinear: u", u)
        assert len(u[0].shape) >= 3, "Shape of u[0] is not 3+ dim"
        assert len(u[1].shape) >= 3, "Shape of u[1] is not 3+ dim"
        assert len(u[2].shape) >= 3, "Shape of u[2] is not 3+ dim"
        u0_shape = u[0].shape
        u1_shape = u[1].shape
        u2_shape = u[2].shape
        if len(u0_shape) == 3:
            u[0].shape = (1, u0_shape[0], u0_shape[1], u0_shape[2])
        #print("navlam_nonlinear: u1_shape", u1_shape)
        if len(u1_shape) == 3:
            #print("navlam_nonlinear: new u1_shape", (1, u1_shape[0], u1_shape[1], u1_shape[2]))
            u[1].shape = (1, u1_shape[0], u1_shape[1], u1_shape[2])
        #print("navlam_nonlinear: u2_shape", u2_shape)
        if len(u2_shape) == 3:
            #print("navlam_nonlinear: new u2_shape", (1, u2_shape[0], u2_shape[1], u2_shape[2]))
            u[2].shape = (1, u2_shape[0], u2_shape[1], u2_shape[2])
        #cells.print_cell("navlam_nonlinear: H", H)

        assert u[0].shape == u[1].shape, "Shape of u[0] and u[1] differ."
        assert u[0].shape == u[2].shape, "Shape of u[0] and u[2] differ."

        nt,nz,ny,nx=u[0].shape

        for i in range(maxniter):

            F[0] = u[0][:,np.r_[1:nz,-1],:,:]*((llambda+2*mu)/H[0,0]) + \
            u[0][:,np.r_[0,:nz-1],:,:]*((llambda+2*mu)/H[0,0]) + \
            u[0][:,:,np.r_[1:ny,-1],:]*(mu/H[1,1]) + \
            u[0][:,:,np.r_[0,:ny-1],:]*(mu/H[1,1]) + \
            u[0][:,:,:,np.r_[1:nx,-1]]*(mu/H[2,2]) + \
            u[0][:,:,:,np.r_[0,:nx-1]]*(mu/H[2,2]) + \
            u[1][:,np.r_[1:nz,-1],np.r_[1:ny,-1],:]*((llambda+mu)/(4*H[0,1])) + \
            u[1][:,np.r_[1:nz,-1],np.r_[0,:ny-1],:]*(-(llambda+mu)/(4*H[0,1])) + \
            u[1][:,np.r_[0,:nz-1],np.r_[1:ny,-1],:]*(-(llambda+mu)/(4*H[0,1])) + \
            u[1][:,np.r_[0,:nz-1],np.r_[0,:ny-1],:]*((llambda+mu)/(4*H[0,1])) + \
            u[2][:,np.r_[1:nz,-1],:,np.r_[1:nx,-1]]*((llambda+mu)/(4*H[0,2])) + \
            u[2][:,np.r_[1:nz,-1],:,np.r_[0,:nx-1]]*(-(llambda+mu)/(4*H[0,2])) + \
            u[2][:,np.r_[0,:nz-1],:,np.r_[1:nx,-1]]*(-(llambda+mu)/(4*H[0,2])) + \
            u[2][:,np.r_[0,:nz-1],:,np.r_[0,:nx-1]]*((llambda+mu)/(4*H[0,2])) - \
            forceu[0]
            F[0] = -F[0]/(-2*mu*(1/H[0,0] + 1/H[1,1] + 1/H[2,2])-2*(llambda+mu)/H[0,0])
        
            F[1] =  u[1][:,np.r_[1:nz,-1],:,:]*(mu/H[0,0]) + \
            u[1][:,np.r_[0,:nz-1],:,:]*(mu/H[0,0]) + \
            u[1][:,:,np.r_[1:ny,-1],:]*((llambda+2*mu)/H[1,1]) + \
            u[1][:,:,np.r_[0,:ny-1],:]*((llambda+2*mu)/H[1,1]) + \
            u[1][:,:,:,np.r_[1:nx,-1]]*(mu/H[2,2]) + \
            u[1][:,:,:,np.r_[0,:nx-1]]*(mu/H[2,2]) + \
            u[0][:,np.r_[1:nz,-1],np.r_[1:ny,-1],:]*((llambda+mu)/(4*H[0,1])) + \
            u[0][:,np.r_[1:nz,-1],np.r_[0,:ny-1],:]*(-(llambda+mu)/(4*H[0,1])) + \
            u[0][:,np.r_[0,:nz-1],np.r_[1:ny,-1],:]*(-(llambda+mu)/(4*H[0,1])) + \
            u[0][:,np.r_[0,:nz-1],np.r_[0,:ny-1],:]*((llambda+mu)/(4*H[0,1])) + \
            u[2][:,:,np.r_[1:ny,-1],np.r_[1:nx,-1]]*((llambda+mu)/(4*H[1,2])) + \
            u[2][:,:,np.r_[1:ny,-1],np.r_[0,:nx-1]]*(-(llambda+mu)/(4*H[1,2])) + \
            u[2][:,:,np.r_[0,:ny-1],np.r_[1:nx,-1]]*(-(llambda+mu)/(4*H[1,2])) + \
            u[2][:,:,np.r_[0,:ny-1],np.r_[0,:nx-1]]*((llambda+mu)/(4*H[1,2])) - \
            forceu[1]
            F[1] = -F[1]/(-2*mu*(1/H[0,0] + 1/H[1,1] + 1/H[2,2])-2*(llambda+mu)/H[1,1])

            F[2] = u[2][:,np.r_[1:nz,-1],:,:]*(mu/H[0,0]) + \
            u[2][:,np.r_[0,:nz-1],:,:]*(mu/H[0,0]) + \
            u[2][:,:,np.r_[1:ny,-1],:]*(mu/H[1,1]) + \
            u[2][:,:,np.r_[0,:ny-1],:]*(mu/H[1,1]) + \
            u[2][:,:,:,np.r_[1:nx,-1]]*((llambda+2*mu)/H[2,2]) + \
            u[2][:,:,:,np.r_[0,:nx-1]]*((llambda+2*mu)/H[2,2]) + \
            u[0][:,np.r_[1:nz,-1],:,np.r_[1:nx,-1]]*((llambda+mu)/(4*H[0,2])) + \
            u[0][:,np.r_[1:nz,-1],:,np.r_[0,:nx-1]]*(-(llambda+mu)/(4*H[0,2])) + \
            u[0][:,np.r_[0,:nz-1],:,np.r_[1:nx,-1]]*(-(llambda+mu)/(4*H[0,2])) + \
            u[0][:,np.r_[0,:nz-1],:,np.r_[0,:nx-1]]*((llambda+mu)/(4*H[0,2])) + \
            u[1][:,:,np.r_[1:ny,-1],np.r_[1:nx,-1]]*((llambda+mu)/(4*H[1,2])) + \
            u[1][:,:,np.r_[1:ny,-1],np.r_[0,:nx-1]]*(-(llambda+mu)/(4*H[1,2])) + \
            u[1][:,:,np.r_[0,:ny-1],np.r_[1:nx,-1]]*(-(llambda+mu)/(4*H[1,2])) + \
            u[1][:,:,np.r_[0,:ny-1],np.r_[0,:nx-1]]*((llambda+mu)/(4*H[1,2])) - \
            forceu[2]
            F[2] = -F[2]/(-2*mu*(1/H[0,0] + 1/H[1,1] + 1/H[2,2])-2*(llambda+mu)/H[2,2])
        
            # fix point iterations
            u[0] = F[0]
            u[1] = F[1]
            u[2] = F[2]

        u[0].shape = u0_shape
        u[1].shape = u1_shape
        u[2].shape = u2_shape
    else:
        raise ValueError("nudim out of range: %d" % prm['nudim'])

    return u
"""


# ----------------

def navlam_nonlinear(forceu, u_in, prm):
    if prm['nudim'] == 2:
        return navlam_nonlinear2(forceu, u_in, prm)
    elif prm['nudim'] == 3:
        return navlam_nonlinear3(forceu, u_in, prm)


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
    cdef Py_ssize_t ny, nx
    cdef int maxniter, i
    cdef size_t j, k

    # prm must contain
    if type(prm['maxniter']) is tuple:
        maxniter = prm['maxniter'][0]
    else:
        maxniter = prm['maxniter']
    h = prm['h']
    cdef int llambda = prm['lambda']
    cdef int mu = prm['mu']
    cdef int dt = prm['dt']
    cdef int nudim = prm['nudim']

    # F = cell(prm['nudim'],1)
    F = {}
    # cdef H ndarray float64
    # cdef DTYPE_t[:, :] H = np.zeros((prm['nudim'],prm['nudim']))
    # cdef DTYPE_t [:, :] H = np.zeros([prm['nudim'],prm['nudim']], dtype=DTYPE)
    assert prm['nudim'] == 2
    cdef DTYPE_t Harr[2][2]
    cdef DTYPE_t[:, :] H = Harr
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


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
@cython.nonecheck(False)
def navlam_nonlinear3(forceu, u_in, prm):
    """
    Fix point iterations (isolating the unknown on left hand side and
    iterating). See page 100 in 'A multigrid tutorial'
    """
    """
    for i = 1 : maxniter
        F{1} = u{1}([2:end end],:,:,:)*((lambda+2*mu)/H(1,1)) + ...
            u{1}([1 1:end-1],:,:,:)*((lambda+2*mu)/H(1,1)) + ...
            u{1}(:,[2:end end],:,:)*(mu/H(2,2)) + ...
            u{1}(:,[1 1:end-1],:,:)*(mu/H(2,2)) + ...
            u{1}(:,:,[2:end end],:)*(mu/H(3,3)) + ...
            u{1}(:,:,[1 1:end-1],:)*(mu/H(3,3)) + ...            
            u{2}([2:end end],[2:end end],:,:)*((lambda+mu)/(4*H(1,2))) + ...
            u{2}([2:end end],[1 1:end-1],:,:)*(-(lambda+mu)/(4*H(1,2))) + ...
            u{2}([1 1:end-1],[2:end end],:,:)*(-(lambda+mu)/(4*H(1,2))) + ...
            u{2}([1 1:end-1],[1 1:end-1],:,:)*((lambda+mu)/(4*H(1,2))) + ...
            u{3}([2:end end],:,[2:end end],:)*((lambda+mu)/(4*H(1,3))) + ...
            u{3}([2:end end],:,[1 1:end-1],:)*(-(lambda+mu)/(4*H(1,3))) + ...
            u{3}([1 1:end-1],:,[2:end end],:)*(-(lambda+mu)/(4*H(1,3))) + ...
            u{3}([1 1:end-1],:,[1 1:end-1],:)*((lambda+mu)/(4*H(1,3))) - ...
            forceu{1};
        F{1} = -F{1}/(-2*mu*(1/H(1,1) + 1/H(2,2) + 1/H(3,3))-2*(lambda+mu)/H(1,1));
        
            
        F{2} =  u{2}([2:end end],:,:,:)*(mu/H(1,1)) + ...
            u{2}([1 1:end-1],:,:,:)*(mu/H(1,1)) + ...
            u{2}(:,[2:end end],:,:)*((lambda+2*mu)/H(2,2)) + ...
            u{2}(:,[1 1:end-1],:,:)*((lambda+2*mu)/H(2,2)) + ...
            u{2}(:,:,[2:end end],:)*(mu/H(3,3)) + ...
            u{2}(:,:,[1 1:end-1],:)*(mu/H(3,3)) + ...           
            u{1}([2:end end],[2:end end],:,:)*((lambda+mu)/(4*H(1,2))) + ...
            u{1}([2:end end],[1 1:end-1],:,:)*(-(lambda+mu)/(4*H(1,2))) + ...
            u{1}([1 1:end-1],[2:end end],:,:)*(-(lambda+mu)/(4*H(1,2))) + ...
            u{1}([1 1:end-1],[1 1:end-1],:,:)*((lambda+mu)/(4*H(1,2))) + ...
            u{3}(:,[2:end end],[2:end end],:)*((lambda+mu)/(4*H(2,3))) + ...
            u{3}(:,[2:end end],[1 1:end-1],:)*(-(lambda+mu)/(4*H(2,3))) + ...
            u{3}(:,[1 1:end-1],[2:end end],:)*(-(lambda+mu)/(4*H(2,3))) + ...
            u{3}(:,[1 1:end-1],[1 1:end-1],:)*((lambda+mu)/(4*H(2,3))) - ...
            forceu{2};
        F{2} = -F{2}/(-2*mu*(1/H(1,1) + 1/H(2,2) + 1/H(3,3))-2*(lambda+mu)/H(2,2));

        
        F{3} = u{3}([2:end end],:,:,:)*(mu/H(1,1)) + ...
            u{3}([1 1:end-1],:,:,:)*(mu/H(1,1)) + ...
            u{3}(:,[2:end end],:,:)*(mu/H(2,2)) + ...
            u{3}(:,[1 1:end-1],:,:)*(mu/H(2,2)) + ...
            u{3}(:,:,[2:end end],:)*((lambda+2*mu)/H(3,3)) +  ...
            u{3}(:,:,[1 1:end-1],:)*((lambda+2*mu)/H(3,3)) + ...           
            u{1}([2:end end],:,[2:end end],:)*((lambda+mu)/(4*H(1,3))) + ...
            u{1}([2:end end],:,[1 1:end-1],:)*(-(lambda+mu)/(4*H(1,3))) + ...
            u{1}([1 1:end-1],:,[2:end end],:)*(-(lambda+mu)/(4*H(1,3))) + ...
            u{1}([1 1:end-1],:,[1 1:end-1],:)*((lambda+mu)/(4*H(1,3))) + ...
            u{2}(:,[2:end end],[2:end end],:)*((lambda+mu)/(4*H(2,3))) + ...
            u{2}(:,[2:end end],[1 1:end-1],:)*(-(lambda+mu)/(4*H(2,3))) + ...
            u{2}(:,[1 1:end-1],[2:end end],:)*(-(lambda+mu)/(4*H(2,3))) + ...
            u{2}(:,[1 1:end-1],[1 1:end-1],:)*((lambda+mu)/(4*H(2,3))) - ...
            forceu{3};
        
        F{3} = -F{3}/(-2*mu*(1/H(1,1) + 1/H(2,2) + 1/H(3,3))-2*(lambda+mu)/H(3,3));
        
        % fix point iterations
        u{1} = F{1};
        u{2} = F{2};
        u{3} = F{3};

                
    end
    """

    assert u_in[0].dtype == DTYPE
    assert u_in[1].dtype == DTYPE
    assert u_in[2].dtype == DTYPE
    assert forceu[0].dtype == DTYPE
    assert forceu[1].dtype == DTYPE
    assert forceu[2].dtype == DTYPE

    # u = u_in.copy() # Do not modify input

    # Py_ssize_t
    # cdef size_t nz, ny, nx, nzend, nyend, nxenc
    cdef Py_ssize_t nz, ny, nx, nzend, nyend, nxend
    cdef int maxniter, i
    # cdef size_t j, k
    cdef Py_ssize_t j, k, iz, iy, ix

    # prm must contain
    if type(prm['maxniter']) is tuple:
        maxniter = prm['maxniter'][0]
    else:
        maxniter = prm['maxniter']
    h = prm['h']
    cdef int llambda = prm['lambda']
    cdef int mu = prm['mu']
    cdef int dt = prm['dt']

    # F = cell(prm['nudim'],1)
    # F = {}
    assert prm['nudim'] == 3
    cdef double Harr[3][3]
    cdef double[:, :] H = Harr
    for j in range(prm['nudim']):
        for k in range(prm['nudim']):
            H[j, k] = h[j] * h[k]

    # cells.print_cell("navlam_nonlinear: u", u)
    assert u_in[0].ndim == 3, "Shape of u_in[0] is not 3 dim, is {} {}".format(u_in[0].ndim, u_in[0].shape)
    assert u_in[1].ndim == 3, "Shape of u_in[1] is not 3 dim"
    assert u_in[2].ndim == 3, "Shape of u_in[2] is not 3 dim"
    u0_shape = u_in[0].shape
    u1_shape = u_in[1].shape
    u2_shape = u_in[2].shape
    for i in range(3):
        if forceu[i].ndim == 4 and forceu[i].shape[0] == 1:
            forceu[i].shape = forceu[i].shape[1:]
    assert forceu[0].ndim == 3, "Shape of forceu[0] is not 3 dim, is {} {}".format(forceu[0].ndim, forceu[0].shape)
    assert forceu[1].ndim == 3, "Shape of forceu[1] is not 3 dim, is {} {}".format(forceu[1].ndim, forceu[1].shape)
    assert forceu[2].ndim == 3, "Shape of forceu[2] is not 3 dim, is {} {}".format(forceu[2].ndim, forceu[2].shape)

    assert u_in[0].shape == u_in[1].shape, "Shape of u_in[0] and u_in[1] differ."
    assert u_in[0].shape == u_in[2].shape, "Shape of u_in[0] and u_in[2] differ."

    nz, ny, nx = u_in[0].shape
    nzend = nz - 1;
    nyend = ny - 1;
    nxend = nx - 1

    cdef DTYPE_t[:, :, :] u0 = (u_in[0])
    cdef DTYPE_t[:, :, :] u1 = (u_in[1])
    cdef DTYPE_t[:, :, :] u2 = (u_in[2])
    cdef DTYPE_t[:, :, :] forceu0 = (forceu[0])
    cdef DTYPE_t[:, :, :] forceu1 = (forceu[1])
    cdef DTYPE_t[:, :, :] forceu2 = (forceu[2])

    cdef DTYPE_t[:, :, :] F0 = np.zeros_like(u_in[0], dtype=DTYPE)
    cdef DTYPE_t[:, :, :] F1 = np.zeros_like(u_in[1], dtype=DTYPE)
    cdef DTYPE_t[:, :, :] F2 = np.zeros_like(u_in[2], dtype=DTYPE)
    cdef DTYPE_t[:, :, :] temp = np.zeros_like(u_in[0], dtype=DTYPE)

    cdef double mu00 = mu / H[0, 0]
    cdef double mu11 = mu / H[1, 1]
    cdef double mu22 = mu / H[2, 2]
    cdef double l2mu00 = (llambda + 2 * mu) / H[0, 0]
    cdef double l2mu11 = (llambda + 2 * mu) / H[1, 1]
    cdef double l2mu22 = (llambda + 2 * mu) / H[2, 2]
    cdef double lm401 = (llambda + mu) / (4 * H[0, 1])
    cdef double nlm401 = -(llambda + mu) / (4 * H[0, 1])
    cdef double lm402 = (llambda + mu) / (4 * H[0, 2])
    cdef double nlm402 = -(llambda + mu) / (4 * H[0, 2])
    cdef double lm412 = (llambda + mu) / (4 * H[1, 2])
    cdef double nlm412 = -(llambda + mu) / (4 * H[1, 2])
    cdef double diag0 = -2 * mu * (1 / H[0, 0] + 1 / H[1, 1] + 1 / H[2, 2]) - 2 * (llambda + mu) / H[0, 0]
    cdef double diag1 = -2 * mu * (1 / H[0, 0] + 1 / H[1, 1] + 1 / H[2, 2]) - 2 * (llambda + mu) / H[1, 1]
    cdef double diag2 = -2 * mu * (1 / H[0, 0] + 1 / H[1, 1] + 1 / H[2, 2]) - 2 * (llambda + mu) / H[2, 2]

    for i in range(maxniter):

        # dz1 = u0[np.r_[1:nz,-1],:,:]*((llambda+2*mu)/H[0,0])
        # F0[0:nz-1,:,:] = u0[1:nz,:,:] * ((llambda+2*mu)/H[0,0])
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F0[iz, iy, ix] = u0[iz + 1, iy, ix] * l2mu00
        # F0[nzend,:,:] = u0[nzend,:,:] * ((llambda+2*mu)/H[0,0])
        for iy in range(ny):
            for ix in range(nx):
                F0[nzend, iy, ix] = u0[nzend, iy, ix] * l2mu00
        # dz2 = u0[np.r_[0,:nz-1],:,:]*((llambda+2*mu)/H[0,0])
        # F0[0,:,:] += u0[0,:,:]*((llambda+2*mu)/H[0,0])
        for iy in range(ny):
            for ix in range(nx):
                F0[0, iy, ix] += u0[0, iy, ix] * l2mu00
        # F0[1:nz,:,:] += u0[:nzend,:,:] * ((llambda+2*mu)/H[0,0])
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F0[iz + 1, iy, ix] += u0[iz, iy, ix] * l2mu00
        # dy1 = u0[:,np.r_[1:ny,-1],:]*(mu/H[1,1])
        # F0[:,0:nyend,:] += u0[:,1:ny,:]*(mu/H[1,1])
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F0[iz, iy, ix] += u0[iz, iy + 1, ix] * mu11
        # F0[:,nyend,:] += u0[:,nyend,:]*(mu/H[1,1])
        for iz in range(nz):
            for ix in range(nx):
                F0[iz, nyend, ix] += u0[iz, nyend, ix] * mu11
        # dy2 = u0[:,np.r_[0,:ny-1],:]*(mu/H[1,1])
        # F0[:,0,:] += u0[:,0,:]*(mu/H[1,1])
        for iz in range(nz):
            for ix in range(nx):
                F0[iz, 0, ix] += u0[iz, 0, ix] * mu11
        # F0[:,1:ny,:] += u0[:,:nyend,:]*(mu/H[1,1])
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F0[iz, iy + 1, ix] += u0[iz, iy, ix] * (mu / H[1, 1])
        # dx1 = u0[:,:,np.r_[1:nx,-1]]*(mu/H[2,2])
        # F0[:,:,0:nxend] += u0[:,:,1:nx]*(mu/H[2,2])
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    F0[iz, iy, ix] += u0[iz, iy, ix + 1] * mu22
        # F0[:,:,nxend] += u0[:,:,nxend]*(mu/H[2,2])
        for iz in range(nz):
            for iy in range(ny):
                F0[iz, iy, nxend] += u0[iz, iy, nxend] * mu22
        # dx2 = u0[:,:,np.r_[0,:nx-1]]*(mu/H[2,2])
        # F0[:,:,0] += u0[:,:,0]*(mu/H[2,2])
        for iz in range(nz):
            for iy in range(ny):
                F0[iz, iy, 0] += u0[iz, iy, 0] * mu22
        # F0[:,:,1:nx] += u0[:,:,:nxend]*(mu/H[2,2])
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    F0[iz, iy, ix + 1] += u0[iz, iy, ix] * mu22

        # temp = u1[:,np.r_[1:ny,-1],:]
        # temp[:,0:nyend,:] = u1[:,1:ny,:]
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    temp[iz, iy, ix] = u1[iz, iy + 1, ix]
        # temp[:,nyend,:] = u1[:,nyend,:]
        for iz in range(nz):
            for ix in range(nx):
                temp[iz, nyend, ix] = u1[iz, nyend, ix]
        # dzy1 = temp[np.r_[1:nz,-1],:,:]*((llambda+mu)/(4*H[0,1]))
        # F0[0:nzend,:,:] += temp[1:nz,:,:]*((llambda+mu)/(4*H[0,1]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F0[iz, iy, ix] += temp[iz + 1, iy, ix] * lm401
        # F0[nzend,:,:] += temp[nzend,:,:]*((llambda+mu)/(4*H[0,1]))
        for iy in range(ny):
            for ix in range(nx):
                F0[nzend, iy, ix] += temp[nzend, iy, ix] * lm401

        # temp = u1[:,np.r_[0,:ny-1],:]
        # temp[:,0,:] = u1[:,0,:]
        for iz in range(nz):
            for ix in range(nx):
                temp[iz, 0, ix] = u1[iz, 0, ix]
        # temp[:,1:ny,:] = u1[:,:nyend,:]
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    temp[iz, iy + 1, ix] = u1[iz, iy, ix]
        # dzy2 = temp[np.r_[1:nz,-1],:,:]*(-(llambda+mu)/(4*H[0,1]))
        # F0[:nzend,:,:] += temp[1:nz,:,:]*(-(llambda+mu)/(4*H[0,1]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F0[iz, iy, ix] += temp[iz + 1, iy, ix] * nlm401
        # F0[nzend,:,:] += temp[nzend,:,:]*(-(llambda+mu)/(4*H[0,1]))
        for iy in range(ny):
            for ix in range(nx):
                F0[nzend, iy, ix] += temp[nzend, iy, ix] * nlm401

        # temp = u1[:,np.r_[1:ny,-1],:]
        # temp[:,0:nyend,:] = u1[:,1:ny,:]
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    temp[iz, iy, ix] = u1[iz, iy + 1, ix]
        # temp[:,nyend,:] = u1[:,nyend,:]
        for iz in range(nz):
            for ix in range(nx):
                temp[iz, nyend, ix] = u1[iz, nyend, ix]
        # dzy3 = temp[np.r_[0,:nz-1],:,:]*(-(llambda+mu)/(4*H[0,1]))
        # F0[0,:,:] += temp[0,:,:]*(-(llambda+mu)/(4*H[0,1]))
        for iy in range(ny):
            for ix in range(nx):
                F0[0, iy, ix] += temp[0, iy, ix] * nlm401
        # F0[1:nz,:,:] += temp[:nzend,:,:]*(-(llambda+mu)/(4*H[0,1]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F0[iz + 1, iy, ix] += temp[iz, iy, ix] * nlm401

        # temp = u1[:,np.r_[0,:ny-1],:]
        # temp[:,0,:] = u1[:,0,:]
        for iz in range(nz):
            for ix in range(nx):
                temp[iz, 0, ix] = u1[iz, 0, ix]
        # temp[:,1:ny,:] = u1[:,:ny-1,:]
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    temp[iz, iy + 1, ix] = u1[iz, iy, ix]
        # dzy4 = temp[np.r_[0,:nz-1],:,:]*((llambda+mu)/(4*H[0,1]))
        # F0[0,:,:] += temp[0,:,:]*((llambda+mu)/(4*H[0,1]))
        for iy in range(ny):
            for ix in range(nx):
                F0[0, iy, ix] += temp[0, iy, ix] * lm401
        # F0[1:nz,:,:] += temp[:nz-1,:,:]*((llambda+mu)/(4*H[0,1]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F0[iz + 1, iy, ix] += temp[iz, iy, ix] * lm401

        # temp = u2[:,:,np.r_[1:nx,-1]]
        # temp[:,:,0:nxend] = u2[:,:,1:nx]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix] = u2[iz, iy, ix + 1]
        # temp[:,:,nxend] = u2[:,:,nxend]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, nxend] = u2[iz, iy, nxend]
        # dzx1 = temp[np.r_[1:nz,-1],:,:]*((llambda+mu)/(4*H[0,2]))
        # F0[:nzend,:,:] += temp[1:nz,:,:]*((llambda+mu)/(4*H[0,2]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F0[iz, iy, ix] += temp[iz + 1, iy, ix] * lm402
        # F0[nzend,:,:] += temp[nzend,:,:]*((llambda+mu)/(4*H[0,2]))
        for iy in range(ny):
            for ix in range(nx):
                F0[nzend, iy, ix] += temp[nzend, iy, ix] * lm402

        # temp = u2[:,:,np.r_[0,:nx-1]]
        # temp[:,:,0] = u2[:,:,0]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, 0] = u2[iz, iy, 0]
        # temp[:,:,1:nx] = u2[:,:,:nx-1]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix + 1] = u2[iz, iy, ix]
        # dzx2 = temp[np.r_[1:nz,-1],:,:]*(-(llambda+mu)/(4*H[0,2]))
        # F0[0:nzend,:,:] += temp[1:nz,:,:]*(-(llambda+mu)/(4*H[0,2]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F0[iz, iy, ix] += temp[iz + 1, iy, ix] * nlm402
        # F0[nzend,:,:] += temp[nzend,:,:]*(-(llambda+mu)/(4*H[0,2]))
        for iy in range(ny):
            for ix in range(nx):
                F0[nzend, iy, ix] += temp[nzend, iy, ix] * nlm402

        # temp = u2[:,:,np.r_[1:nx,-1]]
        # temp[:,:,:nxend] = u2[:,:,1:nx]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix] = u2[iz, iy, ix + 1]
        # temp[:,:,nxend] = u2[:,:,nxend]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, nxend] = u2[iz, iy, nxend]
        # dzx3 = temp[np.r_[0,:nz-1],:,:]*(-(llambda+mu)/(4*H[0,2]))
        # F0[0,:,:] += temp[0,:,:]*(-(llambda+mu)/(4*H[0,2]))
        for iy in range(ny):
            for ix in range(nx):
                F0[0, iy, ix] += temp[0, iy, ix] * nlm402
        # F0[1:nz,:,:] += temp[:nz-1,:,:]*(-(llambda+mu)/(4*H[0,2]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F0[iz + 1, iy, ix] += temp[iz, iy, ix] * nlm402

        # temp = u2[:,:,np.r_[0,:nx-1]]
        # temp[:,:,0] = u2[:,:,0]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, 0] = u2[iz, iy, 0]
        # temp[:,:,1:nx] = u2[:,:,:nx-1]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix + 1] = u2[iz, iy, ix]
        # dzx4 = temp[np.r_[0,:nz-1],:,:]*((llambda+mu)/(4*H[0,2]))
        # F0[0,:,:] += temp[0,:,:]*((llambda+mu)/(4*H[0,2]))
        for iy in range(ny):
            for ix in range(nx):
                F0[0, iy, ix] += temp[0, iy, ix] * lm402
        # F0[1:nz,:,:] += temp[:nz-1,:,:]*((llambda+mu)/(4*H[0,2]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F0[iz + 1, iy, ix] += temp[iz, iy, ix] * lm402
        # F0 = dz1 + dz2 + dy1 + dy2 + dx1 + dx2 + dzy1 + dzy2 + dzy3 + dzy4 + dzx1 + dzx2 + dzx3 + dzx4 - forceu0
        # F0 = -F0/(-2*mu*(1/H[0,0] + 1/H[1,1] + 1/H[2,2])-2*(llambda+mu)/H[0,0])
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    F0[iz, iy, ix] = -(F0[iz, iy, ix] - forceu0[iz, iy, ix]) / diag0

        # ez1 = u1[np.r_[1:nz,-1],:,:]*(mu/H[0,0])
        # F1[0:nzend,:,:] = u1[1:nz,:,:]*(mu/H[0,0])
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F1[iz, iy, ix] = u1[iz + 1, iy, ix] * mu00
        # F1[nzend,:,:] = u1[nzend,:,:]*(mu/H[0,0])
        for iy in range(ny):
            for ix in range(nx):
                F1[nzend, iy, ix] = u1[nzend, iy, ix] * mu00

        # ez2 = u1[np.r_[0,:nz-1],:,:]*(mu/H[0,0])
        # F1[0,:,:] += u1[0,:,:]*(mu/H[0,0])
        for iy in range(ny):
            for ix in range(nx):
                F1[0, iy, ix] += u1[0, iy, ix] * mu00
        # F1[1:nz,:,:] += u1[:nzend,:,:]*(mu/H[0,0])
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F1[iz + 1, iy, ix] += u1[iz, iy, ix] * mu00

        # ey1 = u1[:,np.r_[1:ny,-1],:]*((llambda+2*mu)/H[1,1])
        # F1[:,0:nyend,:] += u1[:,1:ny,:]*((llambda+2*mu)/H[1,1])
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F1[iz, iy, ix] += u1[iz, iy + 1, ix] * l2mu11
        # F1[:,nyend,:] += u1[:,nyend,:]*((llambda+2*mu)/H[1,1])
        for iz in range(nz):
            for ix in range(nx):
                F1[iz, nyend, ix] += u1[iz, nyend, ix] * l2mu11

        # ey2 = u1[:,np.r_[0,:ny-1],:]*((llambda+2*mu)/H[1,1])
        # F1[:,0,:] += u1[:,0,:]*((llambda+2*mu)/H[1,1])
        for iz in range(nz):
            for ix in range(nx):
                F1[iz, 0, ix] += u1[iz, 0, ix] * l2mu11
        # F1[:,1:ny,:] += u1[:,:nyend,:]*((llambda+2*mu)/H[1,1])
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F1[iz, iy + 1, ix] += u1[iz, iy, ix] * l2mu11

        # ex1 = u1[:,:,np.r_[1:nx,-1]]*(mu/H[2,2])
        # F1[:,:,0:nxend] += u1[:,:,1:nx]*(mu/H[2,2])
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    F1[iz, iy, ix] += u1[iz, iy, ix + 1] * mu22
        # F1[:,:,nxend] += u1[:,:,nxend]*(mu/H[2,2])
        for iz in range(nz):
            for iy in range(ny):
                F1[iz, iy, nxend] += u1[iz, iy, nxend] * mu22

        # ex2 = u1[:,:,np.r_[0,:nx-1]]*(mu/H[2,2])
        # F1[:,:,0] += u1[:,:,0]*(mu/H[2,2])
        for iz in range(nz):
            for iy in range(ny):
                F1[iz, iy, 0] += u1[iz, iy, 0] * mu22
        # F1[:,:,1:nx] += u1[:,:,:nxend]*(mu/H[2,2])
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    F1[iz, iy, ix + 1] += u1[iz, iy, ix] * mu22

        # temp = u0[:,np.r_[1:ny,-1],:]
        # temp[:,0:nyend,:] = u0[:,1:ny,:]
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    temp[iz, iy, ix] = u0[iz, iy + 1, ix]
        # temp[:,nyend,:] = u0[:,nyend,:]
        for iz in range(nz):
            for ix in range(nx):
                temp[iz, nyend, ix] = u0[iz, nyend, ix]
        # ezy1 = temp[np.r_[1:nz,-1],:,:]*((llambda+mu)/(4*H[0,1]))
        # F1[0:nzend,:,:] += temp[1:nz,:,:]*((llambda+mu)/(4*H[0,1]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F1[iz, iy, ix] += temp[iz + 1, iy, ix] * lm401
        # F1[nzend,:,:] += temp[nzend,:,:]*((llambda+mu)/(4*H[0,1]))
        for iy in range(ny):
            for ix in range(nx):
                F1[nzend, iy, ix] += temp[nzend, iy, ix] * lm401

        # temp = u0[:,np.r_[0,:ny-1],:]
        # temp[:,0,:] = u0[:,0,:]
        for iz in range(nz):
            for ix in range(nx):
                temp[iz, 0, ix] = u0[iz, 0, ix]
        # temp[:,1:ny,:] = u0[:,:ny-1,:]
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    temp[iz, iy + 1, ix] = u0[iz, iy, ix]
        # ezy2 = temp[np.r_[1:nz,-1],:,:]*(-(llambda+mu)/(4*H[0,1]))
        # F1[0:nzend,:,:] += temp[1:nz,:,:]*(-(llambda+mu)/(4*H[0,1]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F1[iz, iy, ix] += temp[iz + 1, iy, ix] * nlm401
        # F1[nzend,:,:] += temp[nzend,:,:]*(-(llambda+mu)/(4*H[0,1]))
        for iy in range(ny):
            for ix in range(nx):
                F1[nzend, iy, ix] += temp[nzend, iy, ix] * nlm401

        # temp = u0[:,np.r_[1:ny,-1],:]
        # temp[:,:nyend,:] = u0[:,1:ny,:]
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    temp[iz, iy, ix] = u0[iz, iy + 1, ix]
        # temp[:,nyend,:] = u0[:,nyend,:]
        for iz in range(nz):
            for ix in range(nx):
                temp[iz, nyend, ix] = u0[iz, nyend, ix]
        # ezy3 = temp[np.r_[0,:nz-1],:,:]*(-(llambda+mu)/(4*H[0,1]))
        # F1[0,:,:] += temp[0,:,:]*(-(llambda+mu)/(4*H[0,1]))
        for iy in range(ny):
            for ix in range(nx):
                F1[0, iy, ix] += temp[0, iy, ix] * nlm401
        # F1[1:nz,:,:] += temp[:nz-1,:,:]*(-(llambda+mu)/(4*H[0,1]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F1[iz + 1, iy, ix] += temp[iz, iy, ix] * nlm401

        # temp = u0[:,np.r_[0,:ny-1],:]
        # temp[:,0,:] = u0[:,0,:]
        for iz in range(nz):
            for ix in range(nx):
                temp[iz, 0, ix] = u0[iz, 0, ix]
        # temp[:,1:ny,:] = u0[:,:ny-1,:]
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    temp[iz, iy + 1, ix] = u0[iz, iy, ix]
        # ezy4 = temp[np.r_[0,:nz-1],:,:]*((llambda+mu)/(4*H[0,1]))
        # F1[0,:,:] += temp[0,:,:]*((llambda+mu)/(4*H[0,1]))
        for iy in range(ny):
            for ix in range(nx):
                F1[0, iy, ix] += temp[0, iy, ix] * lm401
        # F1[1:nz,:,:] += temp[:nz-1,:,:]*((llambda+mu)/(4*H[0,1]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F1[iz + 1, iy, ix] += temp[iz, iy, ix] * lm401

        # temp = u2[:,:,np.r_[1:nx,-1]]
        # temp[:,:,:nxend] = u2[:,:,1:nx]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix] = u2[iz, iy, ix + 1]
        # temp[:,:,nxend] = u2[:,:,nxend]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, nxend] = u2[iz, iy, nxend]
        # eyx1 = temp[:,np.r_[1:ny,-1],:]*((llambda+mu)/(4*H[1,2]))
        # F1[:,:nyend,:] += temp[:,1:ny,:]*((llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F1[iz, iy, ix] += temp[iz, iy + 1, ix] * lm412
        # F1[:,nyend,:] += temp[:,nyend,:]*((llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for ix in range(nx):
                F1[iz, nyend, ix] += temp[iz, nyend, ix] * lm412

        # temp = u2[:,:,np.r_[0,:nx-1]]
        # temp[:,:,0] = u2[:,:,0]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, 0] = u2[iz, iy, 0]
        # temp[:,:,1:nx] = u2[:,:,:nx-1]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix + 1] = u2[iz, iy, ix]
        # eyx2 = temp[:,np.r_[1:ny,-1],:]*(-(llambda+mu)/(4*H[1,2]))
        # F1[:,:nyend,:] += temp[:,1:ny,:]*(-(llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F1[iz, iy, ix] += temp[iz, iy + 1, ix] * nlm412
        # F1[:,nyend,:] += temp[:,nyend,:]*(-(llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for ix in range(nx):
                F1[iz, nyend, ix] += temp[iz, nyend, ix] * nlm412

        # temp = u2[:,:,np.r_[1:nx,-1]]
        # temp[:,:,:nxend] = u2[:,:,1:nx]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix] = u2[iz, iy, ix + 1]
        # temp[:,:,nxend] = u2[:,:,nxend]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, nxend] = u2[iz, iy, nxend]
        # eyx3 = temp[:,np.r_[0,:ny-1],:]*(-(llambda+mu)/(4*H[1,2]))
        # F1[:,0,:] += temp[:,0,:]*(-(llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for ix in range(nx):
                F1[iz, 0, ix] += temp[iz, 0, ix] * nlm412
        # F1[:,1:ny,:] += temp[:,:ny-1,:]*(-(llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F1[iz, iy + 1, ix] += temp[iz, iy, ix] * nlm412

        # temp = u2[:,:,np.r_[0,:nx-1]]
        # temp[:,:,0] = u2[:,:,0]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, 0] = u2[iz, iy, 0]
        # temp[:,:,1:nx] = u2[:,:,:nx-1]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix + 1] = u2[iz, iy, ix]
        # eyx4 = temp[:,np.r_[0,:ny-1],:]*((llambda+mu)/(4*H[1,2]))
        # F1[:,0,:] += temp[:,0,:]*((llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for ix in range(nx):
                F1[iz, 0, ix] += temp[iz, 0, ix] * lm412
        # F1[:,1:ny,:] += temp[:,:ny-1,:]*((llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F1[iz, iy + 1, ix] += temp[iz, iy, ix] * lm412
        # F1 -= forceu1
        # F1 = -F1/(-2*mu*(1/H[0,0] + 1/H[1,1] + 1/H[2,2])-2*(llambda+mu)/H[1,1])
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    F1[iz, iy, ix] = -(F1[iz, iy, ix] - forceu1[iz, iy, ix]) / diag1

        # fz1 = u2[np.r_[1:nz,-1],:,:]*(mu/H[0,0])
        # F2[0:nzend,:,:] = u2[1:nz,:,:]*(mu/H[0,0])
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F2[iz, iy, ix] = u2[iz + 1, iy, ix] * mu00
        # F2[nzend,:,:] = u2[nzend,:,:]*(mu/H[0,0])
        for iy in range(ny):
            for ix in range(nx):
                F2[nzend, iy, ix] = u2[nzend, iy, ix] * mu00

        # fz2 = u2[np.r_[0,:nz-1],:,:]*(mu/H[0,0])
        # F2[0,:,:] += u2[0,:,:]*(mu/H[0,0])
        for iy in range(ny):
            for ix in range(nx):
                F2[0, iy, ix] += u2[0, iy, ix] * mu00
        # F2[1:nz,:,:] += u2[:nz-1,:,:]*(mu/H[0,0])
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F2[iz + 1, iy, ix] += u2[iz, iy, ix] * mu00

        # fy1 = u2[:,np.r_[1:ny,-1],:]*(mu/H[1,1])
        # F2[:,:nyend,:] += u2[:,1:ny,:]*(mu/H[1,1])
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F2[iz, iy, ix] += u2[iz, iy + 1, ix] * mu11
        # F2[:,nyend,:] += u2[:,nyend,:]*(mu/H[1,1])
        for iz in range(nz):
            for ix in range(nx):
                F2[iz, nyend, ix] += u2[iz, nyend, ix] * mu11

        # fy2 = u2[:,np.r_[0,:ny-1],:]*(mu/H[1,1])
        # F2[:,0,:] += u2[:,0,:]*(mu/H[1,1])
        for iz in range(nz):
            for ix in range(nx):
                F2[iz, 0, ix] += u2[iz, 0, ix] * mu11
        # F2[:,1:ny,:] += u2[:,:ny-1,:]*(mu/H[1,1])
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F2[iz, iy + 1, ix] += u2[iz, iy, ix] * mu11

        # fx1 = u2[:,:,np.r_[1:nx,-1]]*((llambda+2*mu)/H[2,2])
        # F2[:,:,:nxend] += u2[:,:,1:nx]*((llambda+2*mu)/H[2,2])
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    F2[iz, iy, ix] += u2[iz, iy, ix + 1] * l2mu22
        # F2[:,:,nxend] += u2[:,:,nxend]*((llambda+2*mu)/H[2,2])
        for iz in range(nz):
            for iy in range(ny):
                F2[iz, iy, nxend] += u2[iz, iy, nxend] * l2mu22

        # fx2 = u2[:,:,np.r_[0,:nx-1]]*((llambda+2*mu)/H[2,2])
        # F2[:,:,0] += u2[:,:,0]*((llambda+2*mu)/H[2,2])
        for iz in range(nz):
            for iy in range(ny):
                F2[iz, iy, 0] += u2[iz, iy, 0] * l2mu22
        # F2[:,:,1:nx] += u2[:,:,:nx-1]*((llambda+2*mu)/H[2,2])
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    F2[iz, iy, ix + 1] += u2[iz, iy, ix] * l2mu22

        # temp = u0[:,:,np.r_[1:nx,-1]]
        # temp[:,:,:nxend] = u0[:,:,1:nx]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix] = u0[iz, iy, ix + 1]
        # temp[:,:,nxend] = u0[:,:,nxend]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, nxend] = u0[iz, iy, nxend]
        # fzx1 = temp[np.r_[1:nz,-1],:,:]*((llambda+mu)/(4*H[0,2]))
        # F2[:nzend,:,:] += temp[1:nz,:,:]*((llambda+mu)/(4*H[0,2]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F2[iz, iy, ix] += temp[iz + 1, iy, ix] * lm402
        # F2[nzend,:,:] += temp[nzend,:,:]*((llambda+mu)/(4*H[0,2]))
        for iy in range(ny):
            for ix in range(nx):
                F2[nzend, iy, ix] += temp[nzend, iy, ix] * lm402

        # temp = u0[:,:,np.r_[0,:nx-1]]
        # temp[:,:,0] = u0[:,:,0]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, 0] = u0[iz, iy, 0]
        # temp[:,:,1:nx] = u0[:,:,:nx-1]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix + 1] = u0[iz, iy, ix]
        # fzx2 = temp[np.r_[1:nz,-1],:,:]*(-(llambda+mu)/(4*H[0,2]))
        # F2[0:nzend,:,:] += temp[1:nz,:,:]*(-(llambda+mu)/(4*H[0,2]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F2[iz, iy, ix] += temp[iz + 1, iy, ix] * nlm402
        # F2[nzend,:,:] += temp[nzend,:,:]*(-(llambda+mu)/(4*H[0,2]))
        for iy in range(ny):
            for ix in range(nx):
                F2[nzend, iy, ix] += temp[nzend, iy, ix] * nlm402

        # temp = u0[:,:,np.r_[1:nx,-1]]
        # temp[:,:,:nxend] = u0[:,:,1:nx]
        # temp[:,:,nxend] = u0[:,:,nxend]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix] = u0[iz, iy, ix + 1]
                temp[iz, iy, nxend] = u0[iz, iy, nxend]
        # fzx3 = temp[np.r_[0,:nz-1],:,:]*(-(llambda+mu)/(4*H[0,2]))
        # F2[0,:,:] += temp[0,:,:]*(-(llambda+mu)/(4*H[0,2]))
        for iy in range(ny):
            for ix in range(nx):
                F2[0, iy, ix] += temp[0, iy, ix] * nlm402
        # F2[1:nz,:,:] += temp[:nz-1,:,:]*(-(llambda+mu)/(4*H[0,2]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F2[iz + 1, iy, ix] += temp[iz, iy, ix] * nlm402

        # temp = u0[:,:,np.r_[0,:nx-1]]
        # temp[:,:,0] = u0[:,:,0]
        # temp[:,:,1:nx] = u0[:,:,:nx-1]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, 0] = u0[iz, iy, 0]
                for ix in range(nx - 1):
                    temp[iz, iy, ix + 1] = u0[iz, iy, ix]
        # fzx4 = temp[np.r_[0,:nz-1],:,:]*((llambda+mu)/(4*H[0,2]))
        # F2[0,:,:] += temp[0,:,:]*((llambda+mu)/(4*H[0,2]))
        for iy in range(ny):
            for ix in range(nx):
                F2[0, iy, ix] += temp[0, iy, ix] * lm402
        # F2[1:nz,:,:] += temp[:nz-1,:,:]*((llambda+mu)/(4*H[0,2]))
        for iz in range(nz - 1):
            for iy in range(ny):
                for ix in range(nx):
                    F2[iz + 1, iy, ix] += temp[iz, iy, ix] * lm402

        # temp = u1[:,:,np.r_[1:nx,-1]]
        # temp[:,:,:nxend] = u1[:,:,1:nx]
        # temp[:,:,nxend] = u1[:,:,nxend]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix] = u1[iz, iy, ix + 1]
                temp[iz, iy, nxend] = u1[iz, iy, nxend]
        # fyx1 = temp[:,np.r_[1:ny,-1],:]*((llambda+mu)/(4*H[1,2]))
        # F2[:,:nyend,:] += temp[:,1:ny,:]*((llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F2[iz, iy, ix] += temp[iz, iy + 1, ix] * lm412
        # F2[:,nyend,:] += temp[:,nyend,:]*((llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for ix in range(nx):
                F2[iz, nyend, ix] += temp[iz, nyend, ix] * lm412

        # temp = u1[:,:,np.r_[0,:nx-1]]
        # temp[:,:,0] = u1[:,:,0]
        # temp[:,:,1:nx] = u1[:,:,:nx-1]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, 0] = u1[iz, iy, 0]
                for ix in range(nx - 1):
                    temp[iz, iy, ix + 1] = u1[iz, iy, ix]
        # fyx2 = temp[:,np.r_[1:ny,-1],:]*(-(llambda+mu)/(4*H[1,2]))
        # F2[:,:nyend,:] += temp[:,1:ny,:]*(-(llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F2[iz, iy, ix] += temp[iz, iy + 1, ix] * nlm412
        # F2[:,nyend,:] += temp[:,nyend,:]*(-(llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for ix in range(nx):
                F2[iz, nyend, ix] += temp[iz, nyend, ix] * nlm412

        # temp = u1[:,:,np.r_[1:nx,-1]]
        # temp[:,:,:nxend] = u1[:,:,1:nx]
        # temp[:,:,nxend] = u1[:,:,nxend]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx - 1):
                    temp[iz, iy, ix] = u1[iz, iy, ix + 1]
                temp[iz, iy, nxend] = u1[iz, iy, nxend]
        # fyx3 = temp[:,np.r_[0,:ny-1],:]*(-(llambda+mu)/(4*H[1,2]))
        # F2[:,0,:] += temp[:,0,:]*(-(llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for ix in range(nx):
                F2[iz, 0, ix] += temp[iz, 0, ix] * nlm412
        # F2[:,1:ny,:] += temp[:,:ny-1,:]*(-(llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F2[iz, iy + 1, ix] += temp[iz, iy, ix] * nlm412

        # temp = u1[:,:,np.r_[0,:nx-1]]
        # temp[:,:,0] = u1[:,:,0]
        # temp[:,:,1:nx] = u1[:,:,:nx-1]
        for iz in range(nz):
            for iy in range(ny):
                temp[iz, iy, 0] = u1[iz, iy, 0]
                for ix in range(nx - 1):
                    temp[iz, iy, ix + 1] = u1[iz, iy, ix]
        # fyx4 = temp[:,np.r_[0,:ny-1],:]*((llambda+mu)/(4*H[1,2]))
        # F2[:,0,:] += temp[:,0,:]*((llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for ix in range(nx):
                F2[iz, 0, ix] += temp[iz, 0, ix] * lm412
        # F2[:,1:ny,:] += temp[:,:ny-1,:]*((llambda+mu)/(4*H[1,2]))
        for iz in range(nz):
            for iy in range(ny - 1):
                for ix in range(nx):
                    F2[iz, iy + 1, ix] += temp[iz, iy, ix] * lm412

        # F2 = -F2/(-2*mu*(1/H[0,0] + 1/H[1,1] + 1/H[2,2])-2*(llambda+mu)/H[2,2])
        # F2 -= forceu2
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    F2[iz, iy, ix] = -(F2[iz, iy, ix] - forceu2[iz, iy, ix]) / diag2

        # fix point iterations
        # u0 = F0
        # u1 = F1
        # u2 = F2
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    u0[iz, iy, ix] = F0[iz, iy, ix]
                    u1[iz, iy, ix] = F1[iz, iy, ix]
                    u2[iz, iy, ix] = F2[iz, iy, ix]

    # u[0].shape = u0_shape
    # u[1].shape = u1_shape
    # u[2].shape = u2_shape
    # assert u[0].shape == u0_shape, "u[0].shape and u0_shape differ."
    # assert u[1].shape == u1_shape, "u[1].shape and u1_shape differ."
    # assert u[2].shape == u2_shape, "u[2].shape and u2_shape differ."

    return {
        0: np.array(u0),
        1: np.array(u1),
        2: np.array(u2)
    }
