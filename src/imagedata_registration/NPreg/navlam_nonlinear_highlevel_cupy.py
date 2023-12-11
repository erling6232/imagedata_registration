"""navlam_nonlinear_highlevel_cupy"""

import numpy as np
import cupy as cp
# import time


DTYPE = np.float64


# pythran export navlam_nonlinear_highlevel(float64[,,], float64[,,], float64[,,], float64[,,], float64[,,],
#                                           float64[,,], int, float64[3], int, float64, float64, float64)
def navlam_nonlinear_highlevel_cupy(forceu_in0, forceu_in1, forceu_in2, u_in0, u_in1, u_in2, maxniter, h, nudim,
                                    llambda, mu, dt):
    """
    Fix point iterations (isolating the unknown on left hand side and
    iterating). See page 100 in 'A multigrid tutorial'
    """

    print("navlam_nonlinear_highlevel_cupy:")
    print('maxniter: {}'.format(maxniter))

    assert nudim == 3
    H = np.zeros((nudim, nudim))
    for j in range(nudim):
        for k in range(nudim):
            H[j, k] = h[j] * h[k]

    assert u_in0.ndim == 3, "Shape of u[0] is not 3 dim"
    assert u_in1.ndim == 3, "Shape of u[1] is not 3 dim"
    assert u_in2.ndim == 3, "Shape of u[2] is not 3 dim"
    u0_shape = u_in0.shape
    u1_shape = u_in1.shape
    u2_shape = u_in2.shape

    assert u_in0.shape == u_in1.shape, "Shape of u[0] and u[1] differ."
    assert u_in0.shape == u_in2.shape, "Shape of u[0] and u[2] differ."

    nz, ny, nx = u_in0.shape
    nzend = nz - 1
    nyend = ny - 1
    nxend = nx - 1

    u0 = cp.asarray(u_in0)
    u1 = cp.asarray(u_in1)
    u2 = cp.asarray(u_in2)
    forceu0 = cp.asarray(forceu_in0)
    forceu1 = cp.asarray(forceu_in1)
    forceu2 = cp.asarray(forceu_in2)

    F0 = cp.zeros_like(u0, dtype=DTYPE)
    F1 = cp.zeros_like(u1, dtype=DTYPE)
    F2 = cp.zeros_like(u2, dtype=DTYPE)
    temp = cp.zeros_like(u0, dtype=DTYPE)

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

    # print('navlam_nonlinear_highlevel_cupy: clock {}'.format(time.clock() - t))
    # u[0].shape = u0_shape
    # u[1].shape = u1_shape
    # u[2].shape = u2_shape
    assert u0.shape == u0_shape, "u[0].shape and u0_shape differ."
    assert u1.shape == u1_shape, "u[1].shape and u1_shape differ."
    assert u2.shape == u2_shape, "u[2].shape and u2_shape differ."

    if cp.get_array_module(u_in0).__name__ == 'numpy':
        u_out = {0: cp.asnumpy(u0), 1: cp.asnumpy(u1), 2: cp.asnumpy(u2)}
    else:
        u_out = {0: u0, 1: u1, 2: u2}

    return u_out
