"""multilevel"""

import numpy as np
# import math
from .resize import Resize
from .centergrid import centergrid
from .normgrad import normgrad
from .cells import innerprodcell
# import pprint


CYCLE_NONE = 0
CYCLE_V1 = 1
CYCLE_V2 = 2
CYCLE_V3 = 3
CYCLE_W2 = 4
CYCLE_W3 = 5


class Level(object):
    def __init__(self, shape, lvl):
        # Dimensions on this multilevel
        if len(shape) == 2:
            dim = np.array(shape)
            dim[0] = np.ceil(lvl * dim[0])
            dim[1] = np.ceil(lvl * dim[1])
            self.dim = dim
            self.dim3 = dim
        elif len(shape) == 3:
            dim = np.array(shape)
            dim[1] = np.ceil(lvl * dim[1])
            dim[2] = np.ceil(lvl * dim[2])
            self.dim = dim
            self.dim3 = dim
        elif len(shape) == 4:
            dim = np.array(shape)
            dim[2] = np.ceil(lvl * dim[2])
            dim[3] = np.ceil(lvl * dim[3])
            self.dim = dim
            self.dim3 = dim[1:]
        else:
            raise ValueError("Dimension %d out of bounds." % len(shape))

        self.nvox = np.prod(self.dim)
        self.midgrid = self.dim / 2  # np.floor(self.dim/2.)
        self.ext = LevelExt()


class LevelExt(object):
    def __init__(self):
        self.minx = None
        self.maxx = None


class Multilevel(object):
    # def setupmultilevelmg(self, fixed, moving, u):
    def __init__(self, cycle, shape, h, scaling):
        self.cycle = cycle
        self.h = h
        # Solve by multigrid
        if cycle == CYCLE_V1:
            self.nmultilevel = 2
            self.levelSeq = np.array([0, 1, 0])
        elif cycle == CYCLE_V2:
            self.nmultilevel = 3
            self.levelSeq = np.array([0, 1, 2, 1, 0])
        elif cycle == CYCLE_V3:
            self.nmultilevel = 4
            self.levelSeq = np.array([0, 1, 2, 3, 2, 1, 0])
        elif cycle == CYCLE_W2:
            self.nmultilevel = 3
            self.levelSeq = np.array([0, 1, 2, 1, 2, 1, 0])
        elif cycle == CYCLE_W3:
            self.nmultilevel = 4
            self.levelSeq = np.array([0, 1, 2, 3, 2, 3, 2, 1, 2, 3, 2, 3, 2, 1, 0])
        elif cycle == CYCLE_NONE:
            self.nmultilevel = 1
            self.levelSeq = np.array([0])
        else:
            raise ValueError('Value "%s" of self.cycle is unknown.' % cycle)
        self.multilevel = np.zeros(self.nmultilevel)
        for i in range(self.nmultilevel):
            self.multilevel[i] = pow(scaling, i)
        self.multigrid = self.multilevel[self.levelSeq]

        self.level = {}
        # print("Multilevel: self.nmultilevel", self.nmultilevel)
        for m in range(self.nmultilevel):
            self.level[m] = Level(shape, self.multilevel[m])
            # Dimensions on this multilevel
            # self.level[m].ini =  np.zeros(self.level[m].dim,  dtype=np.float32)
            # self.level[m].ini3 = np.zeros(self.level[m].dim3, dtype=np.float32)

            # Pixelsize
            self.level[m].h = self.h / self.multilevel[m]
            if len(self.level[m].h) == 4:
                self.level[m].h[0] = 1

            # Image grid
            self.level[m].x, self.level[m].ext.minx, self.level[m].ext.maxx = \
                centergrid(self.level[m].dim, self.level[m].h)

            # Init images
            self.level[m].fixed = None
            self.level[m].moving = None
            self.level[m].fu = None

    def set_fixed_image(self, fixed):
        # fixed[key][tag,slice,rows,columns]
        for m in range(self.nmultilevel):
            # Resize images
            rsf = Resize(np.array(fixed))
            self.level[m].fixed = rsf.resizeBilinear(self.level[m].dim3)

    def set_moving_image(self, moving):
        # moving[key][tag,slice,rows,columns]
        for m in range(self.nmultilevel):
            # Resize images
            rsm = Resize(np.array(moving))
            self.level[m].moving = rsm.resizeBilinear(self.level[m].dim3)
            self.level[m].fu = self.level[m].moving.copy()

    def set_deformation_field(self, u, ndim):
        # print("set_deformation_field: ndim", ndim)
        # print("set_deformation_field: u"); pprint.pprint(u)
        for m in range(self.nmultilevel):
            self.ndim = ndim
            # Resize images
            # float32: to save memory
            # nt,nz,ny,nx = self.level[m].ini.shape
            self.level[m].u = {}
            # self.level[m].u = np.zeros([self.ndim, nt,nz,ny,nx], dtype=np.float32)
            for i in range(self.ndim):
                if u is None:
                    self.level[m].u[i] = np.zeros(self.level[m].dim, dtype=float)
                else:
                    rsu = Resize(u[i])
                    self.level[m].u[i] = rsu.resizeBilinear(self.level[m].dim).astype(float)
                    # float32: to save memory
        # print("set_deformation_field: self.level[0].u"); pprint.pprint(self.level[0].u)

    def set_gradients(self, nudim, eta):
        self.eta = eta
        for m in range(self.nmultilevel):
            # Find the normalized gradients for later use
            self.level[m].dgn, self.level[m].dg, temp = normgrad(self.level[m].fixed, self.eta, self.level[m].h)
            # for i in self.level[m].dgn.keys():
            self.level[m].absdgn2 = innerprodcell(self.level[m].dgn, self.level[m].dgn)
            absgradg = np.zeros(self.level[m].dim3, dtype=float)

            for j in range(nudim):
                absgradg = absgradg + self.level[m].dg[j] ** 2
            self.level[m].absgradg = np.sqrt(absgradg)

            self.level[m].phi = None
