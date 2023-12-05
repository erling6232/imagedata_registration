"""navlam_nonlinear_3"""

import numpy as np


# from .cells import print_cell

DTYPE = np.float64


# pythran export navlam_nonlinear_3(float64[]+, float64[]+, float64[]+, float64[]+, float64[]+, float64[]+, int,
#                                   float64[3]+, int, float64, float64, float64)
def navlam_nonlinear_3(forceu0, forceu1, forceu2, u0, u1, u2, maxniter, h, nudim, llambda, mu, dt):
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

    assert u0.dtype == DTYPE
    assert u1.dtype == DTYPE
    assert u2.dtype == DTYPE
    assert forceu0.dtype == DTYPE
    assert forceu1.dtype == DTYPE
    assert forceu2.dtype == DTYPE

    # Py_ssize_t
    # cdef Py_ssize_t nz, ny, nx, nzend, nyend, nxenc
    # cdef int maxniter, i
    # cdef Py_ssize_t j, k, iz, iy, ix

    # cdef int llambda = prm['lambda']
    # cdef int mu = prm['mu']
    # cdef int dt = prm['dt']

    # F = cell(prm['nudim'],1)
    # F = {}
    assert nudim == 3
    H = np.zeros((nudim, nudim))
    # cdef double Harr[3][3]
    # cdef double [:, :] H = Harr
    for j in range(nudim):
        for k in range(nudim):
            H[j, k] = h[j] * h[k]

    # cells.print_cell("navlam_nonlinear: u", u)
    assert u0.ndim == 3, "Shape of u[0] is not 3 dim"
    assert u1.ndim == 3, "Shape of u[1] is not 3 dim"
    assert u2.ndim == 3, "Shape of u[2] is not 3 dim"
    u0_shape = u0.shape
    u1_shape = u1.shape
    u2_shape = u2.shape

    assert u0.shape == u1.shape, "Shape of u[0] and u[1] differ."
    assert u0.shape == u2.shape, "Shape of u[0] and u[2] differ."

    nz, ny, nx = u0.shape
    nzend = nz - 1
    nyend = ny - 1
    nxend = nx - 1

    # cdef DTYPE_t [:, :, :] F0 = np.zeros_like(u[0], dtype=DTYPE)
    # cdef DTYPE_t [:, :, :] F1 = np.zeros_like(u[1], dtype=DTYPE)
    # cdef DTYPE_t [:, :, :] F2 = np.zeros_like(u[2], dtype=DTYPE)
    # cdef DTYPE_t [:, :, :] temp = np.zeros_like(u[0], dtype=DTYPE)

    F0 = np.zeros_like(u0, dtype=DTYPE)
    F1 = np.zeros_like(u1, dtype=DTYPE)
    F2 = np.zeros_like(u2, dtype=DTYPE)
    temp = np.zeros_like(u0, dtype=DTYPE)

    # cdef double mu00 = mu/H[0,0]
    # cdef double mu11 = mu/H[1,1]
    # cdef double mu22 = mu/H[2,2]
    # cdef double l2mu00 = (llambda+2*mu)/H[0,0]
    # cdef double l2mu11 = (llambda+2*mu)/H[1,1]
    # cdef double l2mu22 = (llambda+2*mu)/H[2,2]
    # cdef double lm401 = (llambda+mu)/(4*H[0,1])
    # cdef double nlm401 = -(llambda+mu)/(4*H[0,1])
    # cdef double lm402 = (llambda+mu)/(4*H[0,2])
    # cdef double nlm402 = -(llambda+mu)/(4*H[0,2])
    # cdef double lm412 = (llambda+mu)/(4*H[1,2])
    # cdef double nlm412 = -(llambda+mu)/(4*H[1,2])
    # cdef double diag0 = -2*mu*(1/H[0,0] + 1/H[1,1] + 1/H[2,2])-2*(llambda+mu)/H[0,0]
    # cdef double diag1 = -2*mu*(1/H[0,0] + 1/H[1,1] + 1/H[2,2])-2*(llambda+mu)/H[1,1]
    # cdef double diag2 = -2*mu*(1/H[0,0] + 1/H[1,1] + 1/H[2,2])-2*(llambda+mu)/H[2,2]

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
        u0 = F0
        u1 = F1
        u2 = F2

    # u[0].shape = u0_shape
    # u[1].shape = u1_shape
    # u[2].shape = u2_shape
    assert u0.shape == u0_shape, "u[0].shape and u0_shape differ."
    assert u1.shape == u1_shape, "u[1].shape and u1_shape differ."
    assert u2.shape == u2_shape, "u[2].shape and u2_shape differ."

    return u0, u1, u2
