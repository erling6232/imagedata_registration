"""Get cost for elastic registration"""

import numpy as np


def getcost(grid, Eall, prm):
    # Get cost for elastic registration

    # ndim = min(3,prm.ndim);
    # .reg.data = zeros(grid.dim);
    E = {}
    E['reg_data'] = np.zeros(grid.dim, dtype=float)
    E['reg_reg'] = np.zeros(grid.dim, dtype=float)
    E['segm_data'] = np.zeros(grid.dim, dtype=float)
    E['segm_reg'] = np.zeros(grid.dim, dtype=float)

    #
    # Data term registration
    #

    if prm['ssd'] > 0:
        raise ValueError('Cost parameter "ssd" not implemented.')
        # cost = ssdcost(grid.f, grid.g)
        # E['reg_data'] = E['reg_data'] + cost

    if prm['ngf'] > 0:
        # a = grid.fu
        cost = normgradcost(grid.dfun, grid.dgn)
        E['reg_data'] = E['reg_data'] + cost

    """
    if prm['ncp'] > 0:
        a = grid.fu
        cost = normgradcost(grid.fu,grid.g,grid.h,prm.eta)
        E['reg_data'] = E['reg_data'] + cost
    """

    #
    # Regularization registration
    #

    h = grid.h
    if len(h) == 2:
        hz, hy, hx = (1, h[0], h[1])
    elif len(h) == 3:
        hz, hy, hx = (h[0], h[1], h[2])
    elif len(h) == 4:
        hz, hy, hx = (h[1], h[2], h[3])
    du = {}
    for i in range(prm['nudim']):
        du[i] = {}
        (du[i][0], du[i][1], du[i][2]) = derivcentr3(grid.u[i], hz, hy, hx)

    # NB: For these two terms see book Modersiztk page 100 or
    # "Nonrigid image registration Using
    # physically based model" by Zhao Yi, page 17

    # divergence (div u)^2
    a = np.zeros(grid.dim, dtype=float)
    for i in range(prm['nudim']):
        a = a + du[i][i]
    a = a ** 2
    #     a1 = mean(a(:));
    E['reg_reg'] = E['reg_reg'] + (prm['lambda'] / 2) * a

    # (<grade,grade>)^2
    a = np.zeros(grid.dim, dtype=float)
    for i in range(prm['nudim']):
        for j in range(prm['nudim']):
            a = a + (du[i][j] + du[j][i]) ** 2
    #     a = a.^2;
    #     a2 = mean(a(:));
    #     for i = 1 : prm.nudim
    #         for j = 1 : prm.nudim
    #             a = a + (du{i}{j} + du{j}{i}).^2;
    E['reg_reg'] = E['reg_reg'] + (prm['mu'] / 4) * a

    #     a = [a1 a2]

    # summing up the data terms
    Eall['reg_data'].append(E['reg_data'].mean())
    Eall['reg_reg'].append(E['reg_reg'].mean())
    Eall['segm_data'].append(E['segm_data'].mean())
    Eall['segm_reg'].append(E['segm_reg'].mean())
    a = Eall['reg_data'][-1] + \
        Eall['reg_reg'][-1] + \
        Eall['segm_data'][-1] + \
        Eall['segm_reg'][-1]
    Eall['total'].append(a)

    return


def normgradcost(dfn, dgn):
    # compute the cost of normalized gradients

    # normalized gradients
    # [dfn,df,absfreg] = normgrad(f,eta,h);
    # [dgn,dg,absgreg] = normgrad(g,eta,h);

    dim = dfn[0].shape
    ndim = len(dim)
    ndim = min(ndim, 3)
    if len(dim) == 2:
        dim = (1, dim[0], dim[1])
    if len(dim) == 3:
        dim = (1, dim[0], dim[1], dim[2])
    v = np.zeros(dim)
    for i in range(ndim):
        for j in range(dim[0]):
            v[j, :, :, :] = v[j, :, :, :] + dfn[i][j, :, :, :] * dgn[i]
    C = 1 - v ** 2

    return C


def derivcentr3(u, hz, hy, hx):
    """
    % DERIVCENTR3 Finding derivatives
    %
    %   [UZ UY UX] = DERIVCENTR3(U,HZ,HY,HX) Finding the derivatives of U
    %   using a 3 neighbourhood and central differences.
    """

    dim = u.shape
    ndim = len(dim)
    if ndim == 2:
        ny, nx = dim
        ux = (u[:, np.r_[1:nx, -1]] - u[:, np.r_[0, :nx - 1]]) / (2 * hx)
        uy = (u[np.r_[1:ny, -1], :] - u[np.r_[0, :ny - 1], :]) / (2 * hy)
        uz = np.zeros_like(u)
    elif ndim == 3:
        nz, ny, nx = dim
        ux = (u[:, :, np.r_[1:nx, -1]] - u[:, :, np.r_[0, :nx - 1]]) / (2 * hx)
        uy = (u[:, np.r_[1:ny, -1], :] - u[:, np.r_[0, :ny - 1], :]) / (2 * hy)
        uz = (u[np.r_[1:nz, -1], :, :] - u[np.r_[0, :nz - 1], :, :]) / (2 * hz)
    elif ndim == 4:
        nt, nz, ny, nx = dim
        ux = (u[:, :, :, np.r_[1:nx, -1]] - u[:, :, :, np.r_[0, :nx - 1]]) / (2 * hx)
        uy = (u[:, :, np.r_[1:ny, -1], :] - u[:, :, np.r_[0, :ny - 1], :]) / (2 * hy)
        uz = (u[:, np.r_[1:nz, -1], :, :] - u[:, np.r_[0, :nz - 1], :, :]) / (2 * hz)
    else:
        raise ValueError("Shape is out of range.")

    return (uz, uy, ux)
