"""
NPreg image registration.
This is the where the real registration work is done.
"""

from typing import Union
import importlib.metadata
import numpy as np
from .resize import Resize
from .multilevel import (Multilevel,
                         CYCLE_V1, CYCLE_V2, CYCLE_V3, CYCLE_NONE, CYCLE_W2, CYCLE_W3)
# from .centergrid import centergrid
from .transform import TransformLinear
from .gradientreg import gradientreg
from .getcost import getcost


try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "(null)"

try:
    from .multigrid_nonlin_cupy import navlam_nonlinear, multigrid_nonlin
    print('Loaded CUDA multigrid_nonlin_cupy.')
except Exception:
    try:
        from .multigrid_nonlin_cy import navlam_nonlinear, multigrid_nonlin
        print('Loaded compiled multigrid_nonlin_cy.')
    except Exception:
        from .multigrid_nonlin import navlam_nonlinear, multigrid_nonlin
        print('Loaded pure python multigrid_nonlin.py.')
from imagedata.series import Series

MULTIGRID_SOLVER_LINEARFP = 0
MULTIGRID_SOLVER_NONLINEARFP = 1


class NPreg(object):
    """Perform non-parametric image registration of moving to fixed.
    """

    def __init__(self, fixed: Series, prm: dict = {}):
        self.fixed_si = fixed

        # Deformation field
        self.u = None  # np.zeros()

        # Pressure
        self.p = None  # np.zeros()

        # Cost functional to use
        self.ssd = 0 if 'ssd' not in prm else prm['ssd']
        self.ngf = 1 if 'ngf' not in prm else prm['ngf']
        self.ncp = 0 if 'ncp' not in prm else prm['ncp']
        self.ssdngf = 0 if 'ssdngf' not in prm else prm['ssdngf']
        self.ngfparal = 0 if 'ngfparal' not in prm else prm['ngfparal']
        self.mi = 0 if 'mi' not in prm else prm['mi']
        self.hess = 0 if 'hess' not in prm else prm['hess']

        # Edge parameter
        self.eta = 0.03 if 'eta' not in prm else prm['eta']

        # Maximum number of iterations
        self.maxniter = 30 if 'maxniter' not in prm else prm['maxniter']

        # Default solver
        self.multigridsolver = MULTIGRID_SOLVER_NONLINEARFP

        # Regularization
        self.mu = 5 if 'mu' not in prm else prm['mu']
        self.llambda = 5 if 'llambda' not in prm else prm['llambda']

        # Multi-level scaling
        # Note: Always keep at 0.50, otherwise it becomes unstable!!!
        self.scaling = 0.50

        # Voxel size
        # self.h = rowvect(self.fixed_si.spacing)
        # self.h = self.fixed_si.spacing
        # print("NPreg: Remember to restore self.h = self.fixed_si.spacing()")
        # self.h = np.array([1., 1., 1.])
        # print("h", self.h)
        if issubclass(type(fixed), Series):
            self.h = fixed.spacing
        else:
            self.h = np.array([1., 1., 1.])

        # Cycle
        # self.cycle = CYCLE_NONE
        # self.cycle = CYCLE_V1
        self.cycle = CYCLE_V3

        # Regularization segmentation
        self.eps = 0.1 if 'eps' not in prm else prm['eps']

        # Time step
        self.dt = 10 if 'dt' not in prm else prm['dt']

        # Regularization term
        self.alpha = 1 if 'alpha' not in prm else prm['alpha']

        # Pushing phi to -1,1
        self.gamma = 1 if 'gamma' not in prm else prm['gamma']

        # phi
        self.phi = None

        # The masks
        self.mask = None

        self.weight = 0.1 if 'weight' not in prm else prm['weight']

        self.E = {}
        self.E['reg_data'] = []
        self.E['reg_reg'] = []
        self.E['segm_data'] = []
        self.E['segm_reg'] = []
        self.E['total'] = []

        # Normalize fixed image to [0,1]
        self.fixed = self.fixed_si / self.fixed_si.max()

    # def print_to_screen(self):

    def register_volume(self, moving):
        # Normalize moving image to [0,1]
        self.moving = moving / moving.max()

        if len(self.h) != 3:
            raise ValueError("Size of self.h (%d) mismatch. Should be 3." % self.h.size())

        self.nsubphase = 1
        self.nphase = 2

        self.dim = moving.shape
        self.ndim = moving.ndim
        # self.dim = moving.shape
        self.ntime = 1  # volume only
        self.nudim = 3

        # Solve by multigrid
        # Set up multilevel scheme
        self.multi = Multilevel(self.cycle, self.dim, self.h, self.scaling)
        self.multi.set_fixed_image(self.fixed)
        self.multi.set_moving_image(self.moving)
        self.multi.set_deformation_field(self.u, self.ndim)
        self.multi.set_gradients(self.nudim, self.eta)
        self.multigrid = self.multi.multigrid

        # delete fixed; delete moving

        # Print to screen
        # self.print_to_screen()

        # Loop over maximum number of iterations
        self.niter = 0

        print("%35s%14s%13s%13s%13s%13s" % (
            "Multigrid, Iteration # number of %d" % self.maxniter, "Total cost", "Reg,data", "Reg,reg", "Segm,data",
            "Segm,reg"))

        while True:
            # Solve equation by true, nonlinear fixed point iterations
            if self.multigridsolver == MULTIGRID_SOLVER_NONLINEARFP:
                try:
                    self.solve_nonlinearfp()
                except Exception as e:
                    print('solve_nonlinearfp failed: {}'.format(e))
                    raise
            # Solve equation by a linear system, also fixed point iterations
            elif self.multigridsolver == MULTIGRID_SOLVER_LINEARFP:
                raise ValueError("MULTIGRID_SOLVER_LINEARFP is not implemented.")
                # solve_linearfp(self.multi.level)
            else:
                raise ValueError('Value "%s" of self.multigridsolver is unknown.' % self.multigridsolver)

            # make cost functional
            prmin = {}
            prmin['h'] = self.h
            prmin['ssd'] = self.ssd
            prmin['ngf'] = self.ngf
            prmin['nudim'] = self.nudim
            prmin['lambda'] = self.llambda
            prmin['mu'] = self.mu
            # pprint.pprint(self.multi.level[0].dfu[0])
            getcost(self.multi.level[0], self.E, prmin)

            # resize the results
            for i in range(1, self.multi.nmultilevel):
                for j in range(self.ndim):
                    # self.multi.level[0].u[j] = resize(self.multi.level[0].u[0],self.multi.level[i].dim3,'bilinear')
                    rsi = Resize(self.multi.level[0].u[0])
                    self.multi.level[i].u[j] = rsi.resizeBilinear(self.multi.level[i].dim3)

            self.time = min(self.ntime, 1)

            """
            msg = [makestr(['Multigrid, Iteration ' num2str(prm.niter) ' of ' num2str(prm.maxniter)],3*nz) ...
            makestr(num2str(E['total'][prm.niter]),nz) ...
            makestr(num2str(E['reg_data'][prm.niter]),nz),...
            makestr(num2str(E['reg_reg'][prm.niter]),nz),...
            makestr(num2str(E['segm_data'][prm.niter]),nz),...
            makestr(num2str(E['segm_reg'][prm.niter]),nz)];
            disp(msg);
            """

            # print("%39s%13s%13s%13s%13s%13s" % ("Multigrid, # Iteration # number of %d" % self.maxniter,
            #     "Total # cost", "Reg,data", "Reg,reg", "Segm,data", # "Segm,reg"))

            print("Multigrid, Iteration %5d of %5d %13f%13f%13f%13f%13f" % (
                self.niter + 1, self.maxniter, self.E['total'][self.niter], self.E['reg_data'][self.niter],
                self.E['reg_reg'][self.niter], self.E['segm_data'][self.niter], self.E['segm_reg'][self.niter]))

            self.niter += 1
            if self.niter == self.maxniter:
                print("Max number of iterations reached, terminating")
                break

        # apply transform to initial image
        # reg = np.empty_like(moving)
        # self.u = self.multi.level[0].u
        # self.phi = self.multi.level[0].phi
        # reg = applytransform(moving,self.multi.level[0].u[:self.ndim],
        #     self.multi.level[0].x[:self.ndim],self.multi.level[0].ext,'linear')
        transform = TransformLinear(self.multi.level[0].ext)
        out_si = transform.apply(moving, self.ndim, self.multi.level[0].u, self.multi.level[0].x)

        # out_hdr = moving_hdr.copy()
        # return out_hdr, out_si
        return Series(out_si, template=moving, geometry=self.fixed)

    def solve_nonlinearfp(self):

        self.updatef = 0

        # print("solve_nonlinearfp: GRID"); pprint.pprint( self.multi)
        # print(type(self.multi))
        # initialize
        self.multi.level[0].dfu = None  # []
        self.multi.level[0].forceu = {}
        for i in range(self.nudim):
            self.multi.level[0].forceu[i] = np.zeros(tuple(self.multi.level[0].dim))

        # apply transform
        transform = TransformLinear(self.multi.level[0].ext)
        # print("solve_nonlinearfp: len(u)", len(self.multi.level[0].u))
        # print("solve_nonlinearfp: self.multi.level[0].moving "); pprint.pprint(self.multi.level[0].moving)
        # print("solve_nonlinearfp: self.multi.level[0].u"); pprint.pprint(self.multi.level[0].u)
        self.multi.level[0].fu = transform.apply(self.multi.level[0].moving, self.ndim, self.multi.level[0].u,
                                                 self.multi.level[0].x)

        # get gradient for registration using standard cost functionals
        prmin = {}
        prmin['h'] = self.multi.level[0].h
        prmin['ngf'] = self.ngf
        prmin['ssd'] = self.ssd
        prmin['ncp'] = self.ncp
        [self.multi.level[0].forceu,
         self.multi.level[0].dfu,
         self.multi.level[0].dfun,
         Hfun,
         prob] = \
            gradientreg(self.multi.level[0].fu,
                        self.multi.level[0].fixed,
                        self.multi.level[0].dgn,
                        self.multi.level[0].dg,
                        self.multi.level[0].absdgn2,
                        self.eta,
                        self.multi.level[0].h,
                        prmin,
                        None)
        # print("solve_nonlinearfp: self.multi.level[0].dfu "); pprint.pprint(self.multi.level[0].dfu)

        # resize the force on the levels
        b = np.arange(self.nudim)
        for i in range(1, self.multi.nmultilevel):
            for j in range(self.nudim):
                rsi = Resize(self.multi.level[0].forceu[b[j]])
                a = rsi.resizeBilinear(self.multi.level[i].dim3)
                try:
                    self.multi.level[i].forceu[b[j]] = a
                except AttributeError:
                    self.multi.level[i].forceu = {}
                    self.multi.level[i].forceu[b[j]] = a

        # print("solve_nonlinearfp: self.cycle", self.cycle)
        if self.cycle == CYCLE_V1:
            self.maxnitersolver = (10, 50, 10)
        elif self.cycle == CYCLE_V2:
            self.maxnitersolver = (5, 10, 30, 10, 5)
        elif self.cycle == CYCLE_V3:
            self.maxnitersolver = (5, 10, 15, 30, 15, 10, 5)
        elif self.cycle == CYCLE_W2:
            self.maxnitersolver = (10, 20, 500, 20, 500, 20, 10)
        elif self.cycle == CYCLE_W3:
            self.maxnitersolver = (10, 20, 30, 500, 30, 500, 30, 20, 30, 500, 30, 500, 30, 20, 10)
        elif self.cycle == CYCLE_NONE:
            self.maxnitersolver = (100,)
            # solve linear system without multigrid
            prmin['h'] = self.multi.level[0].h
            prmin['dim'] = self.multi.level[0].dim
            prmin['multigrid'] = self.multigrid
            prmin['maxniter'] = self.maxnitersolver
            prmin['level'] = self.multi.levelSeq
            prmin['nudim'] = self.nudim
            prmin['lambda'] = self.llambda
            prmin['mu'] = self.mu
            prmin['dt'] = self.dt
            # print("solve_nonlinearfp: before navlam_nonlinear u")
            # pprint.pprint(self.multi.level[0].u)
            # 3#t = time.clock()
            # 3# tt = navlam_nonlinear_3(self.multi.level[0].forceu,self.multi.level[0].u,prmin)
            # 3# self.multi.level[0].u = navlam_nonlinear_3(self.multi.level[0].forceu,self.multi.level[0].u,prmin)
            # 3# print('navlam_nonlinear_3: clock {}'.format(time.clock()-t))
            # 3#tt = navlam_nonlinear_cc(self.multi.level[0].forceu,self.multi.level[0].u,prmin)
            # 3#print('navlam_nonlinear_cc3: clock {}'.format(time.clock()-t))
            # t = time.clock()
            self.multi.level[0].u = navlam_nonlinear(self.multi.level[0].forceu, self.multi.level[0].u, prmin)
            # print('multigrid_nonlin_cupy.navlam_nonlinear: clock {}'.format(time.clock()-t))
            # t = time.clock()
            # tt = navlam_nonlinear_highlevel_cupy(
            #    self.multi.level[0].forceu[0],
            #    self.multi.level[0].forceu[1],
            #    self.multi.level[0].forceu[2],
            #    self.multi.level[0].u[0],
            #    self.multi.level[0].u[1],
            #    self.multi.level[0].u[2],
            #    prmin['maxniter'][0],
            #    prmin['h'],
            #    prmin['nudim'],
            #    float(prmin['lambda']),
            #    float(prmin['mu']),
            #    float(prmin['dt'])
            # )
            # print('navlam_nonlinear_highlevel_cupy: clock {}'.format(time.clock()-t))
            # std#t = time.clock()
            # std#self.multi.level[0].u = navlam_nonlinear(self.multi.level[0].forceu,self.multi.level[0].u,prmin)
            # std#print('navlam_nonlinear: clock {}'.format(time.clock()-t))
            # print("solve_nonlinearfp: after navlam_nonlinear u")
            # pprint.pprint(self.multi.level[0].u)

            # self.multi.level = self.multi.level[0]
            # return self.multi.level
            return

        # v = cell(self.multi.nmultilevel,1)
        # r = cell(self.multi.nmultilevel,1)
        # e = cell(self.multi.nmultilevel,1)
        # for i = 1 : self.multi.nmultilevel:
        #     v{i} = self.multi.level{i}.u
        # u = v

        # run multigrid
        # forceu = cell(self.multi.nmultilevel,1)
        # u = cell(self.multi.nmultilevel,1)
        # h = cell(self.multi.nmultilevel,1)
        # dim = cell(self.multi.nmultilevel,1)
        forceu = {}
        u = {}
        h = {}
        dim = {}
        for i in range(self.multi.nmultilevel):
            forceu[i] = self.multi.level[i].forceu
            u[i] = self.multi.level[i].u
            h[i] = self.multi.level[i].h
            dim[i] = self.multi.level[i].dim
        prmin['h'] = h
        prmin['dim'] = dim
        prmin['multigrid'] = self.multigrid
        prmin['maxniter'] = self.maxnitersolver
        # print("npreg: self.multi.level", self.multi.level)
        # print("npreg: self.multi.level.level", self.multi.levelSeq)
        prmin['level'] = self.multi.levelSeq
        prmin['nudim'] = self.nudim
        prmin['lambda'] = self.llambda
        prmin['mu'] = self.mu
        prmin['dt'] = self.dt

        # 3# t = time.clock()
        # 3# self.multi.level[0].u = multigrid_nonlin_3(forceu,u,prmin)
        # 3# print('navlam_nonlinear_3: clock {}'.format(time.clock()-t))
        # 3# t = time.clock()
        self.multi.level[0].u = multigrid_nonlin(forceu, u, prmin)
        # 3# print('navlam_nonlinear: clock {}'.format(time.clock()-t))
        # GRID = GRID[0]

        # return GRID


def register_npreg(
        fixed: Union[int, Series],
        moving: Series,
        cycle: int = CYCLE_NONE,
        prm: dict = {}) -> Series:
    """Register a series using NPreg.

    Args:
        fixed (int or Series): Fixed volume, or index into moving
        moving (Series): Moving volume(s)
        cycle (int): Cycle
    Returns:
        Registered series (Series)
    """

    if issubclass(type(fixed), int):
        fixed = moving[fixed]
    # dim = np.array(fixed.shape)
    # h = fixed.spacing
    # ext = LevelExt()
    # x, ext.minx, ext.maxx = centergrid(dim, h)

    if moving.ndim > fixed.ndim:
        shape = (moving.shape[0],) + fixed.shape
        tags = moving.tags[0]
    else:
        shape = fixed.shape
        tags = [None]
    si = np.zeros(shape, dtype=moving.dtype)

    for t, tag in enumerate(tags):
        print('-------------------------------------------------')
        print('NPreg register {} of {}'.format(t + 1, len(tags)))
        npreg = NPreg(fixed, prm=prm)
        npreg.cycle = cycle
        # print("test_register_volume: moving", type(si_moving), si_moving.dtype, si_moving.shape)
        # print("test_register_volume: moving", type(moving), moving.dtype, moving.shape)
        if tag is None:
            si = npreg.register_volume(moving)
        else:
            si[t] = npreg.register_volume(moving[t])
        # print('------DONE-{}-------------------------------------'.format(time.clock()-t0))
        print('------DONE----------------------------------------')

    super_threshold_indices = si > 65500
    si[super_threshold_indices] = 0

    out = Series(si, input_order=moving.input_order, template=moving, geometry=fixed)
    if out.ndim > fixed.ndim:
        out.tags = moving.tags
        out.axes = out.axes._replace(**{out.input_order: moving.axes[0]})
    try:
        out.seriesDescription += " NPreg"
    except ValueError:
        out.seriesDescription = "NPreg"

    return out
