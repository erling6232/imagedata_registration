"""multigrid_nonlin_cupy"""

# import time
import numpy as np
import cupy as cp
# from .cells import print_cell
from .resize import Resize


# from .navlam_nonlinear_highlevel_cupy import navlam_nonlinear_3
# from .navlam_nonlinear_3 import navlam_nonlinear_3
from .navlam_nonlinear_highlevel_cupy import navlam_nonlinear_highlevel_cupy


DTYPE = np.float64


def multigrid_nonlin(forceu, u_in, prm):
    print('multigrid_nonlin:')
    u = u_in.copy()  # Do not modify input

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

    interpmethod = 'bilinear'

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
        li = level[i]
        prmin['maxniter'] = maxniter[i]

        # make coarser
        if i < nlevel - 1 and multigrid[i] > multigrid[i + 1]:
            ln = level[i + 1]

            # solve equation
            # print("multigrid_nonlin: l", l)
            prmin['h'] = h[li]
            if prm['nudim'] == 2:
                u[li] = navlam_nonlinear(forceu[li], v[li], prmin)
            elif prm['nudim'] == 3:
                # u[l] = navlam_nonlinear_3(
                # u[l] = navlam_nonlinear_highlevel_cupy(
                u[li] = navlam_nonlinear_highlevel_cupy(
                    forceu[li][0], forceu[li][1], forceu[li][2],
                    v[li][0].copy(), v[li][1].copy(), v[li][2].copy(),
                    prmin['maxniter'], prmin['h'], prmin['nudim'],
                    prmin['lambda'], prmin['mu'], prmin['dt'])
            v[li] = u[li]

            # find Av
            av = Au(v[li], h[li], prmin)

            # find residual r
            for j in range(noptdim):
                if li not in r:
                    r[li] = {}
                r[li][a[j]] = forceu[li][a[j]] - av[a[j]]

            # restrict
            for j in range(noptdim):
                # r[ln][a[j]] = resize(r[l][a[j]], dim3[ln], interpmethod)
                rsi = Resize(r[li][a[j]])
                if ln not in r:
                    r[ln] = {}
                r[ln][a[j]] = rsi.resize(dim3[ln], interpmethod)

            for j in range(noptdim):
                # v[ln][a[j]] = resize(v[l][a[j]], dim3[ln], interpmethod)
                rsi = Resize(v[li][a[j]])
                if ln not in v:
                    v[ln] = {}
                v[ln][a[j]] = rsi.resize(dim3[ln], interpmethod)
            continue

        # at the bottom, solve equation
        if multigrid[i] < multigrid[i - 1] and multigrid[i] < multigrid[i + 1]:

            # find Au
            prmin['h'] = h[li]
            av = Au(v[li], h[li], prmin)

            # find new RHS
            for j in range(prm['nudim']):
                forceu[li][a[j]] = av[a[j]] + r[li][a[j]]

            # solve equation
            prmin['h'] = h[li]
            if prm['nudim'] == 2:
                u[li] = navlam_nonlinear(forceu[li], v[li], prmin)
            elif prm['nudim'] == 3:
                # u[l] = navlam_nonlinear_3(
                # u[l] = navlam_nonlinear_highlevel_cupy(
                u[li] = navlam_nonlinear(
                    forceu[li][0], forceu[li][1], forceu[li][2],
                    v[li][0].copy(), v[li][1].copy(), v[li][2].copy(),
                    prmin['maxniter'], prmin['h'], prmin['nudim'],
                    prmin['lambda'], prmin['mu'], prmin['dt'])

            # find error e
            for j in range(prm['nudim']):
                if li not in e:
                    e[li] = {}
                e[li][a[j]] = u[li][a[j]] - v[li][a[j]]
            continue

        # refine and correct
        if multigrid[i] > multigrid[i - 1] and i > 1:
            lp = level[i - 1]

            # find error e
            for j in range(prm['nudim']):
                # e[l][a[j]] = resize(e[lp][a[j]], dim3[l], interpmethod)
                rsi = Resize(e[lp][a[j]])
                if li not in e:
                    e[li] = {}
                e[li][a[j]] = rsi.resize(dim3[li], interpmethod)

            # correct v by e
            for j in range(prm['nudim']):
                v[li][a[j]] = v[li][a[j]] + e[li][a[j]]

            # relax with initial guess v
            prmin['h'] = h[li]
            if prm['nudim'] == 2:
                u[li] = navlam_nonlinear(forceu[li], v[li], prmin)
            elif prm['nudim'] == 3:
                # u[l] = navlam_nonlinear_3(
                # u[l] = navlam_nonlinear_highlevel_cupy(
                u[li] = navlam_nonlinear(
                    forceu[li][0], forceu[li][1], forceu[li][2],
                    v[li][0].copy(), v[li][1].copy(), v[li][2].copy(),
                    prmin['maxniter'], prmin['h'], prmin['nudim'],
                    prmin['lambda'], prmin['mu'], prmin['dt'])
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
    """
    Fix point iterations (isolating the unknown on left hand side and
    iterating). See page 100 in 'A multigrid tutorial'
    """

    # u = u_in.copy()  # Do not modify input
    u = {}
    for key in u_in.keys():
        u[key] = cp.asarray(u_in[key])

    # prm must contain
    if type(prm['maxniter']) is tuple:
        maxniter = prm['maxniter'][0]
    else:
        maxniter = prm['maxniter']
    h = prm['h']
    llambda = prm['lambda']
    mu = prm['mu']
    # dt = prm['dt']

    # F = cell(prm['nudim'],1)
    F = {}
    H = np.zeros((prm['nudim'], prm['nudim']))
    for j in range(prm['nudim']):
        for k in range(prm['nudim']):
            H[j, k] = h[j] * h[k]

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

        u_out = {0: u[0], 1: u[1]}
        u_out[0].shape = u0_shape
        u_out[1].shape = u1_shape

    elif prm['nudim'] == 3:

        u0 = cp.asarray(u_in[0])
        u1 = cp.asarray(u_in[1])
        u2 = cp.asarray(u_in[2])

        # cells.print_cell("navlam_nonlinear: u", u)
        assert len(u0.shape) >= 3, "Shape of u[0] is not 3+ dim"
        assert len(u1.shape) >= 3, "Shape of u[1] is not 3+ dim"
        assert len(u2.shape) >= 3, "Shape of u[2] is not 3+ dim"
        u0_shape = u0.shape
        u1_shape = u1.shape
        u2_shape = u2.shape
        # cells.print_cell("navlam_nonlinear: H", H)

        assert u0.shape == u1.shape, "Shape of u[0] and u[1] differ."
        assert u0.shape == u2.shape, "Shape of u[0] and u[2] differ."

        for i in range(3):
            assert u[i].shape[-3:] == forceu[i].shape[-3:], (
                "Shape of u[{}] and forceu[{}] differ.".format(i, i))

        nz, ny, nx = u0.shape
        nzend = nz - 1
        nyend = ny - 1
        nxend = nx - 1

        F0 = cp.zeros_like(u0, dtype=DTYPE)
        F1 = cp.zeros_like(u1, dtype=DTYPE)
        F2 = cp.zeros_like(u2, dtype=DTYPE)
        temp = cp.zeros_like(u0, dtype=DTYPE)

        forceu0 = cp.asarray(forceu[0])
        forceu1 = cp.asarray(forceu[1])
        forceu2 = cp.asarray(forceu[2])

        mu00 = mu / H[0, 0]
        mu11 = mu / H[1, 1]
        mu22 = mu / H[2, 2]
        l2mu00 = (llambda + 2 * mu) / H[0, 0]
        l2mu11 = (llambda + 2 * mu) / H[1, 1]
        l2mu22 = (llambda + 2 * mu) / H[2, 2]
        lm401 = (llambda + mu) / (4 * H[0, 1])
        nlm401 = -(llambda + mu) / (4 * H[0, 1])
        lm402 = (llambda + mu) / (4 * H[0, 2])
        nlm402 = -(llambda + mu) / (4 * H[0, 2])
        lm412 = (llambda + mu) / (4 * H[1, 2])
        nlm412 = -(llambda + mu) / (4 * H[1, 2])
        diag0 = -2 * mu * (1 / H[0, 0] + 1 / H[1, 1] + 1 / H[2, 2]) - 2 * (llambda + mu) / H[0, 0]
        diag1 = -2 * mu * (1 / H[0, 0] + 1 / H[1, 1] + 1 / H[2, 2]) - 2 * (llambda + mu) / H[1, 1]
        diag2 = -2 * mu * (1 / H[0, 0] + 1 / H[1, 1] + 1 / H[2, 2]) - 2 * (llambda + mu) / H[2, 2]

        # t = time.clock()

        for i in range(maxniter):
            # dz1 = u0[np.r_[1:nz,-1],:,:]*((llambda+2*mu)/H[0,0])
            F0[0:nz - 1, :, :] = u0[1:nz, :, :] * l2mu00
            F0[nzend, :, :] = u0[nzend, :, :] * l2mu00
            # dz2 = u0[np.r_[0,:nz-1],:,:]*((llambda+2*mu)/H[0,0])
            F0[0, :, :] += u0[0, :, :] * l2mu00
            F0[1:nz, :, :] += u0[:nzend, :, :] * l2mu00
            # dy1 = u0[:,np.r_[1:ny,-1],:]*(mu/H[1,1])
            F0[:, 0:nyend, :] += u0[:, 1:ny, :] * mu11
            F0[:, nyend, :] += u0[:, nyend, :] * mu11
            # dy2 = u0[:,np.r_[0,:ny-1],:]*(mu/H[1,1])
            F0[:, 0, :] += u0[:, 0, :] * mu11
            F0[:, 1:ny, :] += u0[:, :nyend, :] * mu11
            # dx1 = u0[:,:,np.r_[1:nx,-1]]*(mu/H[2,2])
            F0[:, :, 0:nxend] += u0[:, :, 1:nx] * mu22
            F0[:, :, nxend] += u0[:, :, nxend] * mu22
            # dx2 = u0[:,:,np.r_[0,:nx-1]]*(mu/H[2,2])
            F0[:, :, 0] += u0[:, :, 0] * mu22
            F0[:, :, 1:nx] += u0[:, :, :nxend] * mu22

            # temp = u1[:,np.r_[1:ny,-1],:]
            temp[:, 0:nyend, :] = u1[:, 1:ny, :]
            temp[:, nyend, :] = u1[:, nyend, :]
            # dzy1 = temp[np.r_[1:nz,-1],:,:]*((llambda+mu)/(4*H[0,1]))
            F0[0:nzend, :, :] += temp[1:nz, :, :] * lm401
            F0[nzend, :, :] += temp[nzend, :, :] * lm401

            # temp = u1[:,np.r_[0,:ny-1],:]
            temp[:, 0, :] = u1[:, 0, :]
            temp[:, 1:ny, :] = u1[:, :nyend, :]
            # dzy2 = temp[np.r_[1:nz,-1],:,:]*(-(llambda+mu)/(4*H[0,1]))
            F0[:nzend, :, :] += temp[1:nz, :, :] * nlm401
            F0[nzend, :, :] += temp[nzend, :, :] * nlm401

            # temp = u1[:,np.r_[1:ny,-1],:]
            temp[:, 0:nyend, :] = u1[:, 1:ny, :]
            temp[:, nyend, :] = u1[:, nyend, :]
            # dzy3 = temp[np.r_[0,:nz-1],:,:]*(-(llambda+mu)/(4*H[0,1]))
            F0[0, :, :] += temp[0, :, :] * nlm401
            F0[1:nz, :, :] += temp[:nzend, :, :] * nlm401

            # temp = u1[:,np.r_[0,:ny-1],:]
            temp[:, 0, :] = u1[:, 0, :]
            temp[:, 1:ny, :] = u1[:, :ny - 1, :]
            # dzy4 = temp[np.r_[0,:nz-1],:,:]*((llambda+mu)/(4*H[0,1]))
            F0[0, :, :] += temp[0, :, :] * lm401
            F0[1:nz, :, :] += temp[:nz - 1, :, :] * lm401

            # temp = u2[:,:,np.r_[1:nx,-1]]
            temp[:, :, 0:nxend] = u2[:, :, 1:nx]
            temp[:, :, nxend] = u2[:, :, nxend]
            # dzx1 = temp[np.r_[1:nz,-1],:,:]*((llambda+mu)/(4*H[0,2]))
            F0[:nzend, :, :] += temp[1:nz, :, :] * lm402
            F0[nzend, :, :] += temp[nzend, :, :] * lm402

            # temp = u2[:,:,np.r_[0,:nx-1]]
            temp[:, :, 0] = u2[:, :, 0]
            temp[:, :, 1:nx] = u2[:, :, :nx - 1]
            # dzx2 = temp[np.r_[1:nz,-1],:,:]*(-(llambda+mu)/(4*H[0,2]))
            F0[0:nzend, :, :] += temp[1:nz, :, :] * nlm402
            F0[nzend, :, :] += temp[nzend, :, :] * nlm402

            # temp = u2[:,:,np.r_[1:nx,-1]]
            temp[:, :, :nxend] = u2[:, :, 1:nx]
            temp[:, :, nxend] = u2[:, :, nxend]
            # dzx3 = temp[np.r_[0,:nz-1],:,:]*(-(llambda+mu)/(4*H[0,2]))
            F0[0, :, :] += temp[0, :, :] * nlm402
            F0[1:nz, :, :] += temp[:nz - 1, :, :] * nlm402

            # temp = u2[:,:,np.r_[0,:nx-1]]
            temp[:, :, 0] = u2[:, :, 0]
            temp[:, :, 1:nx] = u2[:, :, :nx - 1]
            # dzx4 = temp[np.r_[0,:nz-1],:,:]*((llambda+mu)/(4*H[0,2]))
            F0[0, :, :] += temp[0, :, :] * lm402
            F0[1:nz, :, :] += temp[:nz - 1, :, :] * lm402
            # F0 = dz1 + dz2 + dy1 + dy2 + dx1 + dx2 + dzy1 + dzy2 + dzy3 + dzy4 + dzx1 + dzx2 + dzx3 + dzx4 - forceu0
            F0 -= forceu0
            F0 = -F0 / diag0

            # ez1 = u1[np.r_[1:nz,-1],:,:]*(mu/H[0,0])
            F1[0:nzend, :, :] = u1[1:nz, :, :] * mu00
            F1[nzend, :, :] = u1[nzend, :, :] * mu00

            # ez2 = u1[np.r_[0,:nz-1],:,:]*(mu/H[0,0])
            F1[0, :, :] += u1[0, :, :] * mu00
            F1[1:nz, :, :] += u1[:nzend, :, :] * mu00

            # ey1 = u1[:,np.r_[1:ny,-1],:]*((llambda+2*mu)/H[1,1])
            F1[:, 0:nyend, :] += u1[:, 1:ny, :] * l2mu11
            F1[:, nyend, :] += u1[:, nyend, :] * l2mu11

            # ey2 = u1[:,np.r_[0,:ny-1],:]*((llambda+2*mu)/H[1,1])
            F1[:, 0, :] += u1[:, 0, :] * l2mu11
            F1[:, 1:ny, :] += u1[:, :nyend, :] * l2mu11

            # ex1 = u1[:,:,np.r_[1:nx,-1]]*(mu/H[2,2])
            F1[:, :, 0:nxend] += u1[:, :, 1:nx] * mu22
            F1[:, :, nxend] += u1[:, :, nxend] * mu22

            # ex2 = u1[:,:,np.r_[0,:nx-1]]*(mu/H[2,2])
            F1[:, :, 0] += u1[:, :, 0] * mu22
            F1[:, :, 1:nx] += u1[:, :, :nxend] * mu22

            # temp = u0[:,np.r_[1:ny,-1],:]
            temp[:, 0:nyend, :] = u0[:, 1:ny, :]
            temp[:, nyend, :] = u0[:, nyend, :]
            # ezy1 = temp[np.r_[1:nz,-1],:,:]*((llambda+mu)/(4*H[0,1]))
            F1[0:nzend, :, :] += temp[1:nz, :, :] * lm401
            F1[nzend, :, :] += temp[nzend, :, :] * lm401

            # temp = u0[:,np.r_[0,:ny-1],:]
            temp[:, 0, :] = u0[:, 0, :]
            temp[:, 1:ny, :] = u0[:, :ny - 1, :]
            # ezy2 = temp[np.r_[1:nz,-1],:,:]*(-(llambda+mu)/(4*H[0,1]))
            F1[0:nzend, :, :] += temp[1:nz, :, :] * nlm401
            F1[nzend, :, :] += temp[nzend, :, :] * nlm401

            # temp = u0[:,np.r_[1:ny,-1],:]
            temp[:, :nyend, :] = u0[:, 1:ny, :]
            temp[:, nyend, :] = u0[:, nyend, :]
            # ezy3 = temp[np.r_[0,:nz-1],:,:]*(-(llambda+mu)/(4*H[0,1]))
            F1[0, :, :] += temp[0, :, :] * nlm401
            F1[1:nz, :, :] += temp[:nz - 1, :, :] * nlm401

            # temp = u0[:,np.r_[0,:ny-1],:]
            temp[:, 0, :] = u0[:, 0, :]
            temp[:, 1:ny, :] = u0[:, :ny - 1, :]
            # ezy4 = temp[np.r_[0,:nz-1],:,:]*((llambda+mu)/(4*H[0,1]))
            F1[0, :, :] += temp[0, :, :] * lm401
            F1[1:nz, :, :] += temp[:nz - 1, :, :] * lm401

            # temp = u2[:,:,np.r_[1:nx,-1]]
            temp[:, :, :nxend] = u2[:, :, 1:nx]
            temp[:, :, nxend] = u2[:, :, nxend]
            # eyx1 = temp[:,np.r_[1:ny,-1],:]*((llambda+mu)/(4*H[1,2]))
            F1[:, :nyend, :] += temp[:, 1:ny, :] * lm412
            F1[:, nyend, :] += temp[:, nyend, :] * lm412

            # temp = u2[:,:,np.r_[0,:nx-1]]
            temp[:, :, 0] = u2[:, :, 0]
            temp[:, :, 1:nx] = u2[:, :, :nx - 1]
            # eyx2 = temp[:,np.r_[1:ny,-1],:]*(-(llambda+mu)/(4*H[1,2]))
            F1[:, :nyend, :] += temp[:, 1:ny, :] * nlm412
            F1[:, nyend, :] += temp[:, nyend, :] * nlm412

            # temp = u2[:,:,np.r_[1:nx,-1]]
            temp[:, :, :nxend] = u2[:, :, 1:nx]
            temp[:, :, nxend] = u2[:, :, nxend]
            # eyx3 = temp[:,np.r_[0,:ny-1],:]*(-(llambda+mu)/(4*H[1,2]))
            F1[:, 0, :] += temp[:, 0, :] * nlm412
            F1[:, 1:ny, :] += temp[:, :ny - 1, :] * nlm412

            # temp = u2[:,:,np.r_[0,:nx-1]]
            temp[:, :, 0] = u2[:, :, 0]
            temp[:, :, 1:nx] = u2[:, :, :nx - 1]
            # eyx4 = temp[:,np.r_[0,:ny-1],:]*((llambda+mu)/(4*H[1,2]))
            F1[:, 0, :] += temp[:, 0, :] * lm412
            F1[:, 1:ny, :] += temp[:, :ny - 1, :] * lm412
            F1 -= forceu1
            F1 = -F1 / diag1

            # fz1 = u2[np.r_[1:nz,-1],:,:]*(mu/H[0,0])
            F2[0:nzend, :, :] = u2[1:nz, :, :] * mu00
            F2[nzend, :, :] = u2[nzend, :, :] * mu00

            # fz2 = u2[np.r_[0,:nz-1],:,:]*(mu/H[0,0])
            F2[0, :, :] += u2[0, :, :] * mu00
            F2[1:nz, :, :] += u2[:nz - 1, :, :] * mu00

            # fy1 = u2[:,np.r_[1:ny,-1],:]*(mu/H[1,1])
            F2[:, :nyend, :] += u2[:, 1:ny, :] * mu11
            F2[:, nyend, :] += u2[:, nyend, :] * mu11

            # fy2 = u2[:,np.r_[0,:ny-1],:]*(mu/H[1,1])
            F2[:, 0, :] += u2[:, 0, :] * mu11
            F2[:, 1:ny, :] += u2[:, :ny - 1, :] * mu11

            # fx1 = u2[:,:,np.r_[1:nx,-1]]*((llambda+2*mu)/H[2,2])
            F2[:, :, :nxend] += u2[:, :, 1:nx] * l2mu22
            F2[:, :, nxend] += u2[:, :, nxend] * l2mu22

            # fx2 = u2[:,:,np.r_[0,:nx-1]]*((llambda+2*mu)/H[2,2])
            F2[:, :, 0] += u2[:, :, 0] * l2mu22
            F2[:, :, 1:nx] += u2[:, :, :nx - 1] * l2mu22

            # temp = u0[:,:,np.r_[1:nx,-1]]
            temp[:, :, :nxend] = u0[:, :, 1:nx]
            temp[:, :, nxend] = u0[:, :, nxend]
            # fzx1 = temp[np.r_[1:nz,-1],:,:]*((llambda+mu)/(4*H[0,2]))
            F2[:nzend, :, :] += temp[1:nz, :, :] * lm402
            F2[nzend, :, :] += temp[nzend, :, :] * lm402

            # temp = u0[:,:,np.r_[0,:nx-1]]
            temp[:, :, 0] = u0[:, :, 0]
            temp[:, :, 1:nx] = u0[:, :, :nx - 1]
            # fzx2 = temp[np.r_[1:nz,-1],:,:]*(-(llambda+mu)/(4*H[0,2]))
            F2[0:nzend, :, :] += temp[1:nz, :, :] * nlm402
            F2[nzend, :, :] += temp[nzend, :, :] * nlm402

            # temp = u0[:,:,np.r_[1:nx,-1]]
            temp[:, :, :nxend] = u0[:, :, 1:nx]
            temp[:, :, nxend] = u0[:, :, nxend]
            # fzx3 = temp[np.r_[0,:nz-1],:,:]*(-(llambda+mu)/(4*H[0,2]))
            F2[0, :, :] += temp[0, :, :] * nlm402
            F2[1:nz, :, :] += temp[:nz - 1, :, :] * nlm402

            # temp = u0[:,:,np.r_[0,:nx-1]]
            temp[:, :, 0] = u0[:, :, 0]
            temp[:, :, 1:nx] = u0[:, :, :nx - 1]
            # fzx4 = temp[np.r_[0,:nz-1],:,:]*((llambda+mu)/(4*H[0,2]))
            F2[0, :, :] += temp[0, :, :] * lm402
            F2[1:nz, :, :] += temp[:nz - 1, :, :] * lm402

            # temp = u1[:,:,np.r_[1:nx,-1]]
            temp[:, :, :nxend] = u1[:, :, 1:nx]
            temp[:, :, nxend] = u1[:, :, nxend]
            # fyx1 = temp[:,np.r_[1:ny,-1],:]*((llambda+mu)/(4*H[1,2]))
            F2[:, :nyend, :] += temp[:, 1:ny, :] * lm412
            F2[:, nyend, :] += temp[:, nyend, :] * lm412

            # temp = u1[:,:,np.r_[0,:nx-1]]
            temp[:, :, 0] = u1[:, :, 0]
            temp[:, :, 1:nx] = u1[:, :, :nx - 1]
            # fyx2 = temp[:,np.r_[1:ny,-1],:]*(-(llambda+mu)/(4*H[1,2]))
            F2[:, :nyend, :] += temp[:, 1:ny, :] * nlm412
            F2[:, nyend, :] += temp[:, nyend, :] * nlm412

            # temp = u1[:,:,np.r_[1:nx,-1]]
            temp[:, :, :nxend] = u1[:, :, 1:nx]
            temp[:, :, nxend] = u1[:, :, nxend]
            # fyx3 = temp[:,np.r_[0,:ny-1],:]*(-(llambda+mu)/(4*H[1,2]))
            F2[:, 0, :] += temp[:, 0, :] * nlm412
            F2[:, 1:ny, :] += temp[:, :ny - 1, :] * nlm412

            # temp = u1[:,:,np.r_[0,:nx-1]]
            temp[:, :, 0] = u1[:, :, 0]
            temp[:, :, 1:nx] = u1[:, :, :nx - 1]
            # fyx4 = temp[:,np.r_[0,:ny-1],:]*((llambda+mu)/(4*H[1,2]))
            F2[:, 0, :] += temp[:, 0, :] * lm412
            F2[:, 1:ny, :] += temp[:, :ny - 1, :] * lm412

            F2 -= forceu2
            F2 = -F2 / diag2

            # fix point iterations
            u0[:] = F0[:]
            u1[:] = F1[:]
            u2[:] = F2[:]

        # print('navlam_nonlinear end: clock {}'.format(time.clock()-t))
        if cp.get_array_module(u_in[0]).__name__ == 'cupy':
            u_out = {0: u0, 1: u1, 2: u2}
        else:
            u_out = {0: cp.asnumpy(u0), 1: cp.asnumpy(u1), 2: cp.asnumpy(u2)}
        u_out[0].shape = u0_shape
        u_out[1].shape = u1_shape
        u_out[2].shape = u2_shape
    else:
        raise ValueError("nudim out of range: %d" % prm['nudim'])

    return u_out
