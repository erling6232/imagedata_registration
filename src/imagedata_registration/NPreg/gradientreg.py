"""gradientreg"""

import numpy as np
from .normgrad import normgrad
from .cells import (hessiancell, multconstcell, divideconstcell, sumcell, matrixprodcell,
                    innerprodcell)  # , print_cell
# import pprint


"""
function [grad,dfu,dfun,Hfun,prob] = gradientreg(fu,g,dgn,dg,absdgn2,eta,h,prmin,opts)
"""


def gradientreg(fu, g, dgn, dg, absdgn2, eta, h, prmin, opts):
    """
    print("gradientreg: fu", fu.shape, fu.dtype)
    pprint.pprint(fu)
    print("gradientreg: g", g.shape, g.dtype)
    pprint.pprint(g)
    print("gradientreg: dg", len(dg))
    pprint.pprint(dg)
    print("gradientreg: absdgn2", absdgn2.shape, absdgn2.dtype)
    pprint.pprint(absdgn2)
    print("gradientreg: h ", h.shape, h.dtype, h)
    """
    prm = {}
    prm['ssd'] = 0
    prm['ngf'] = 1
    prm['ngfparal'] = 0
    prm['hess'] = 0
    prm['ngfheavi'] = 0
    prm['ssdngf'] = 0
    prm['ngfabs'] = 0
    prm['ncp'] = 0
    prm['mi'] = 0
    prm['hessabs'] = 0
    # prm = mergestruct(prm,prmin);
    prm = dict(prm, **prmin)

    shape = fu.shape  # Save original shape of fu
    dim = shape
    ndim = len(dim)
    nudim = min(ndim, 3)
    if len(dim) == 2:
        rows, columns = dim
        dim = (1, 1, rows, columns)
        ntime = 1
    elif len(dim) == 3:
        slices, rows, columns = dim
        dim = (1, slices, rows, columns)
        ntime = 1
    if len(dim) != 4:
        raise ValueError("Dimension %d out of bounds." % len(dim))
    # dim3 = dim[1:]
    ntime = dim[0]
    fu.shape = dim

    #
    # Find the normalized gradients
    #
    # ini = np.zeros(dim, dtype=float);

    #
    # Normalized gradients
    #

    # initialize to save time
    # dfun = cell(nudim,1);
    # dfu = cell(nudim,1);
    dfun = {}
    dfu = {}
    for i in range(nudim):
        dfun[i] = np.zeros(dim, dtype=float)
        dfu[i] = np.zeros(dim, dtype=float)
    absfureg = np.zeros(dim, dtype=float)

    for i in range(ntime):
        # do this in 3D since we dont want temporal changes; i.e. we dont want
        # the registration to change the temporal information, so the
        # intensities flow across the timepoints, only spatially!
        # print("gradientreg: ndim", ndim, ", len(h[-ndim:])", len(h[-ndim:]))
        [dfuno, dfuo, absfureg[i, :, :, :]] = normgrad(fu[i, :, :, :], eta, h[-ndim:])
        # print("gradientreg: fu", fu.shape, fu.dtype)
        # a,b,c = normgrad(fu[i,:,:,:],eta,h[-ndim:])
        # print("normgrad", i)
        # [dfuno,dfuo,absfureg[i,:,:,:]] = a,b,c
        # print_cell("gradientreg: dfuno", dfuno)
        # print_cell("gradientreg: dfun", dfun)
        # print_cell("gradientreg: dfu ", dfu)
        # print_cell("gradientreg: dfuo", dfuo)
        for j in range(nudim):
            dfun[j][i, :, :, :] = dfuno[j]
            dfu[j][i, :, :, :] = dfuo[j]
    # print("dfun"); pprint.pprint(dfun)
    # print("dfu "); pprint.pprint(dfu)
    # [dgn,dg,absgreg] = normgrad(g,eta,h(-ndim:))
    # print("gradientreg: absfureg", absfureg.shape)
    # pprint.pprint(absfureg)

    # initialize the gradients in case no method is specified
    # grad = cell(nudim,1);
    grad = {}
    for i in range(nudim):
        grad[i] = np.zeros(dim, dtype=float)

    #
    # SSD
    #
    if prm['ssd'] > 0:
        # kssd = cell(nudim,1);
        kssd = {}
        for i in range(nudim):
            kssd[i] = np.zeros(dim, dtype=float)
            for j in range(ntime):
                kssd[i][j, :, :, :] = (fu[j, :, :, :] - g) * dfu[i][j, :, :, :]
            # add up
            # NB shall we plus here!
            grad[i] = grad[i] + prm['ssd'] * kssd[i]

    #
    # NGF
    #
    if prm['ngf'] > 0:
        # intialize
        # kngf = cell(nudim,1);
        kngf = {}
        for i in range(3):
            kngf[i] = np.zeros(dim, dtype=float)

        for i in range(ntime):
            # dfunhere = cell(nudim,1);
            dfunhere = {}
            for j in range(nudim):
                dfunhere[j] = dfun[j][i, :, :, :]

            # 3D gradient, not 4D on purpose since we dont want to travel information over time points!
            gradhere = gradngf(fu[i, :, :, :], dfunhere, dgn, absfureg[i, :, :, :], h[-3:])
            # print_cell("gradientreg: gradhere", gradhere)
            # pprint.pprint(gradhere)
            for j in range(nudim):
                kngf[j][i, :, :, :] = gradhere[j][0]
        for i in range(nudim):
            # add up
            grad[i] = grad[i] + prm['ngf'] * kngf[i]

    #
    # NGF heaviside
    #

    """
    # NB not working
    if prm['ngfheavi'] > 0:
        # initialize
        k = {}
        for i in range(3):
            k{i} = np.zeros(dim, dtype=float)
        for i = range(ntime):
            # 3D gradient, not 4D on purpose since we dont want to travel information over time points!         
            gradhere = gradngfheavi(fu[i,:,:,:],dfun,dfu,dgn,dg,absfureg,h[-3:])

            for j in range(nudim):
                k[j][i,:,:,:] = gradhere[j]
        for i in range(nudim):
            # add up
            grad[i] = grad[i] + prm['ngfheavi']*k[i]        
    """

    #
    # NGF abs
    #
    """
    if prm.ngfabs > 0
        % intialize
        k = cell(nudim,1);
        for i = 1 : nudim
            k{i} = np.zeros(dim, dtype=float)
        end;        
        for i = 1 : ntime        
                
            % 3D gradient, not 4D on purpose since we dont want to travel information over time points!
            gradhere = gradngfabs(fu(:,:,:,i),g,h(1:3));
                               
            for j = 1 : nudim
                k{j,1}(:,:,:,i) = gradhere{j};
            end;
        end;    
        for i = 1 : nudim
            % add up
            grad{i} = grad{i} + prm.ngfabs*k{i};        
        end;
    end;

    #
    # HESS abs
    #
    if prm.hessabs > 0
        % intialize
        k = cell(nudim,1);
        for i = 1 : 3
            k{i} = np.zeros(dim, dtype=float)
        end;        
        for i = 1 : ntime        
                    
            % 3D gradient, not 4D on purpose since we dont want to travel information over time points!
            gradhere = gradhessabs(fu(:,:,:,i),g,h(1:3));
                                   
            for j = 1 : nudim
                k{j,1}(:,:,:,i) = gradhere{j};
            end;
        end;    
        for i = 1 : nudim
            % add up
            grad{i} = grad{i} + prm.hessabs*k{i};        
        end;
    end;

    #
    # NCP (Cross product)
    #
    if prm.ncp > 0
        % intialize
        k = cell(nudim,1);
        for i = 1 : nudim
            k{i} = np.zeros(dim, dtype=float)
        end;        
        for i = 1 : ntime        
            dfunhere = cell(nudim,1);        
            for j = 1 : nudim
                dfunhere{j,1} = dfun{j}(:,:,:,i);            
            end;
                   
            % 3D gradient, not 4D on purpose since we dont want to travel information over time points!
            gradhere = gradncp(fu(:,:,:,i),dfunhere,dgn,absfureg(:,:,:,i),absdgn2,h(1:nudim));
                                
            for j = 1 : nudim
                k{j,1}(:,:,:,i) = gradhere{j};
            end;
        end;    
        for i = 1 : nudim
            % add up
            grad{i} = grad{i} + prm.ncp*k{i};        
        end;    
    end;

    #
    # Hessian force field
    #
    Hfun = None
    if prm.hess > 0
        % initialize
        k = cell(nudim,1);
        Hfun = cell(nudim*nudim,1);
        for i = 1 : ntime
            [gradhere,Hfuno] = gradhess(fu(:,:,:,i),opts.Hgn,h,eta,ndim);
            for j = 1 : nudim
                k{j,1}(:,:,:,i) = gradhere{j};
            end;        
            for j = 1 : nudim*nudim
                Hfun{j,1}(:,:,:,i) = Hfuno{j};
            end;    
        end;
        for i = 1 : nudim
            % add up
            grad{i} = grad{i} + prm.hess*k{i};        
        end;
    end;


    #
    # NGF same direction of vectors
    #
    if prm.ngfparal == 1
        % intialize
        kngfparal = cell(nudim,1);
        for i = 1 : nudim
            kngfparal{i} = np.zeros(dim, dtype=float)
        end;
    
        for i = 1 : ntime        
        
            dfun = cell(nudim,1);        
            for j = 1 : nudim
                dfun{j,1} = dfun{j}(:,:,:,i);            
            end;
    
            % 3D gradient, not 4D on purpose since we dont want to travel information over time points!
            gradhere = gradngfparal(fu(:,:,:,i),dfun,opts.dgn,absfureg(:,:,:,i),h(1:3));    
            for j = 1 : nudim
                kngfparal{j,1}(:,:,:,i) = gradhere{j};
            end;
        end;    
        for i = 1 : nudim
            % add up
            grad{i} = grad{i} + prm.ngfparal*kngfparal{i};        
        end;
    end;


    #
    # SSDNGF
    #
    if prm.ssdngf > 0
        % intialize
        k = cell(nudim,1);
        for i = 1 : nudim
            k{i} = np.zeros(dim, dtype=float)
        end;        
        for i = 1 : ntime        
                
            % 3D gradient, not 4D on purpose since we dont want to travel information over time points!
            gradhere = gradssdngf(fu(:,:,:,i),g,eta,h(1:3));
                               
            for j = 1 : nudim
                k{j,1}(:,:,:,i) = gradhere{j};
            end;
        end;    
        for i = 1 : nudim
            % add up
            grad{i} = grad{i} + prm.ssdngf*k{i};        
        end;
    end;


    #
    # MI
    #
    prob = None
    if prm.mi > 0
        kmi = cell(nudim,1);
        for i = 1 : ntime
         
            dfuin = cell(nudim,1);        
            for j = 1 : nudim
                dfuin{j,1} = dfu{j}(:,:,:,i);            
            end;
            [gradhere,probout] = gradmi(fu(:,:,:,i),dfuin,g);
            prob.p12(:,:,:,i) = probout.p12;
            prob.P1(:,:,:,i) = probout.P1;
            prob.P2(:,:,:,i) = probout.P2;
            for j = 1 : nudim
                kmi{j}(:,:,:,i) = gradhere{j};            
            end;
        
        end;
        for i = 1 : nudim
            % add up
            grad{i} = grad{i} + prm.mi*kmi{i};        
        end;    
    end;
    """

    fu.shape = shape  # Restore original shape of fu
    for i in range(nudim):
        grad[i].shape = shape
    Hfun = None
    prob = None
    return grad, dfu, dfun, Hfun, prob


def gradngf(fu, dfun, dgn, absfureg, h):
    #
    # The Euler-Lagrange equations
    #
    # print_cell("gradngf: dgn ", dgn)
    # print_cell("gradngf: h ", h); print(h)
    # print_cell("gradngf: fu ", fu); pprint.pprint(fu)

    # Hessian matrix
    # H = hessiancell(fu,h,'ind',ind,'df',dfu);
    H = hessiancell(fu, h)
    # print_cell("gradngf: H   ", H)
    # pprint.pprint(H)

    # dC/du = (grad(f)*grad(g)/||grad(f)||)H(f)*(grad(g) - (grad(f)*grad(g))*grad(f))
    # NB normalized gradients here: grad(f)/||grad(f)||
    # print_cell("gradngf: dfun", dfun)
    # print_cell("gradngf: dgn ", dgn)
    s = innerprodcell(dfun, dgn)
    # print_cell("gradngf: s   ", s)

    # (grad(f)^T * grad(g))*H
    part1 = multconstcell(H, 2 * s)
    # print_cell("gradngf: etter multconstcell part1", part1)

    # divide by regularized absolute F
    # print_cell("gradngf: for   divideconstcell absfureg", absfureg)
    # pprint.pprint(absfureg)
    part1 = divideconstcell(part1, absfureg)
    # print_cell("gradngf: etter divideconstcell part1", part1)

    # -(gradf*gradg)*gradf
    part2 = multconstcell(dfun, -s)
    # print_cell("gradngf: etter multconstcell part2", part2)

    # add grad(gn)
    part2 = sumcell(part2, dgn)
    # print_cell("gradngf: etter sumcell part2", part2)

    # multiply the two parts
    gradngf = matrixprodcell(part1, part2)
    # print_cell("gradngf: etter matrixprodcell gradngf", gradngf)

    # minus in front
    gradngf = multconstcell(gradngf, -1)
    # print_cell("gradngf: etter multconstcell gradngf", gradngf)

    return gradngf


"""
def gradngf(fu,dfun,dgn,absfureg,h):
    #
    # Alternative approach without the cell functions
    #

    # Hessian matrix
    H = hessiancell(fu,h)
    
    prm.ndim = size(H,1)
    
    innerprod = zeros(size(fu))
    for i = 1 : prm.ndim
        innerprod = innerprod + dfun{i} .* dgn{i}
    # innerprod = innerprodcell(fu,g)

    part1 = cell(prm.ndim,prm.ndim)
    c = 2*innerprod./absfureg
    for i = 1 : prm.ndim
        for j = 1 : prm.ndim
            part1{i,j} = H{i,j} .* c

    part2 = cell(prm.ndim,1)
    for i = 1 : prm.ndim
        part2{i,1} = dgn{i} - innerprod .* dfun{i}

    kngf = cell(prm.ndim,1)
    for i = 1 : prm.ndim
        kngf{i,1} = zeros(size(fu))
        for j = 1 : prm.ndim
            kngf{i,1} = kngf{i,1} + part1{i,j} .* part2{j}

    return gradngf
"""
