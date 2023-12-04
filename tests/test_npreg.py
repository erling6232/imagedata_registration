#!/usr/bin/env python3

import unittest
import numpy as np
import pprint
from imagedata.series import Series

from src.imagedata_registration.NPreg import register_npreg
from src.imagedata_registration.NPreg.resize import Resize
from src.imagedata_registration.NPreg.centergrid import centergrid
from src.imagedata_registration.NPreg.translate_image import translate_image
from src.imagedata_registration.NPreg.normgrad import normgrad
from src.imagedata_registration.NPreg import NPreg
from src.imagedata_registration.NPreg.transform import TransformLinear
from src.imagedata_registration.NPreg.cells import ctransposecell, innerprodcell
from src.imagedata_registration.NPreg.multilevel import Level, LevelExt, Multilevel, CYCLE_NONE, CYCLE_V1, CYCLE_V2, CYCLE_V3, CYCLE_W2, CYCLE_W3


class TestResizeFunctions(unittest.TestCase):

    def test_resize(self):
        x = np.linspace(1, 4, 11)
        y = np.linspace(4, 7, 22)
        z = np.linspace(7, 9, 33)
        V = np.zeros((11, 22, 33))
        for i in range(11):
            for j in range(22):
                for k in range(33):
                    V[i, j, k] = 100 * x[i] + 10 * y[j] + z[k]
        rsi = Resize(V)
        self.assertEqual(rsi.resizeNearest((2, 2, 2)).shape, (2, 2, 2))
        self.assertEqual(rsi.resizeBilinear((2, 2, 2)).shape, (2, 2, 2))
        self.assertEqual(rsi.resizeQubic((2, 2, 2)).shape, (2, 2, 2))

        self.assertAlmostEqual(rsi.resizeNearest((2, 2, 2)).mean(), 313., places=7)
        self.assertAlmostEqual(rsi.resizeBilinear((2, 2, 2)).mean(), 313., places=7)
        self.assertAlmostEqual(rsi.resizeQubic((2, 2, 2)).mean(), 313., places=7)
        # with self.assertRaises(ZeroDivisionError):
        #    average([])
        # with self.assertRaises(TypeError):
        #    average(20, 30, 70)


class TestGrid(unittest.TestCase):
    def test_centergrid(self):
        dim = np.array([3, 4, 4], dtype=int)
        h = np.array([1., 1., 1.])
        x, minx, maxx = centergrid(dim, h)
        np.testing.assert_array_equal(x[0], np.array(
            [[[-2., -2., -2., -2.], [-2., -2., -2., -2.], [-2., -2., -2., -2.], [-2., -2., -2., -2.]],
             [[-1., -1., -1., -1.], [-1., -1., -1., -1.], [-1., -1., -1., -1.], [-1., -1., -1., -1.]],
             [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]]))
        np.testing.assert_array_equal(maxx, np.array([0., 0.5, 0.5]))
        np.testing.assert_array_equal(minx, np.array([-2., -2.5, -2.5]))


def create_linear_3d_matrix(shape):
    if len(shape) != 3:
        raise ValueError("Shape is not 3D")
    f = np.zeros(shape)
    i = 0
    for slice in range(f.shape[0]):
        for row in range(f.shape[1]):
            for column in range(f.shape[2]):
                f[slice, row, column] = i
                i += 1
    return f


class TestTransform(unittest.TestCase):
    def test_linear_simple(self):
        dim = np.array([2, 3, 4], dtype=int)
        h = np.array([1., 1., 1.])
        ext = LevelExt()
        x, ext.minx, ext.maxx = centergrid(dim, h)
        transform = TransformLinear(ext)
        moving = create_linear_3d_matrix([2, 3, 4])
        u = {}
        u[0] = np.zeros([2, 3, 4])
        u[1] = np.zeros([2, 3, 4])
        u[2] = np.ones([2, 3, 4])
        fu = transform.apply(moving, 3, u, x)
        cfu = np.array([[[1., 2., 3., 3.], [5., 6., 7., 7.], [9., 10., 11., 11.]],
                        [[13., 14., 15., 15.], [17., 18., 19., 19.], [21., 22., 23., 23.]]])
        np.testing.assert_array_almost_equal(fu, cfu, decimal=2)


class TestCells(unittest.TestCase):
    def test_ctransposecell(self):
        a = {};
        i = 0
        for r in range(3):
            a[r] = {}
            for c in range(3):
                a[r][c] = i
                i += 1
        b = ctransposecell(a)
        ca = {};
        i = 0
        for r in range(3):
            ca[r] = {}
        for c in range(3):
            for r in range(3):
                ca[r][c] = i
                i += 1
        np.testing.assert_array_equal(b, ca)

    def test_innerprodcell_vector(self):
        dfn = {}
        dfn[0] = np.zeros([2, 3, 4])
        dfn[1] = np.zeros([2, 3, 4])
        dfn[2] = np.ones([2, 3, 4])
        dgn = {}
        dgn[0] = np.zeros([2, 3, 4])
        dgn[1] = np.zeros([2, 3, 4])
        dgn[2] = np.ones([2, 3, 4])
        s = innerprodcell(dfn, dgn)
        cs = np.array([[[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                       [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]])
        np.testing.assert_array_almost_equal(s, cs, decimal=4)

    def test_innerprodcell_matrix(self):
        dfn = {}
        dgn = {}
        n = 1
        for i in range(2):
            dfn[i] = {}
            dgn[i] = {}
            for j in range(2):
                dfn[i][j] = np.zeros([2, 3, 4])
                dfn[i][j][0, :, :] = np.eye(3, 4) * n
                dfn[i][j][1, :, :] = np.eye(3, 4) * n
                dgn[i][j] = np.zeros([2, 3, 4])
                dgn[i][j][0, :, :] = np.eye(3, 4) * n
                dgn[i][j][1, :, :] = np.eye(3, 4) * n
                n += 1
        s = innerprodcell(dfn, dgn)
        cs = np.array([[[30, 0, 0, 0], [0, 30, 0, 0], [0, 0, 30, 0]], [[30, 0, 0, 0], [0, 30, 0, 0], [0, 0, 30, 0]]])
        np.testing.assert_array_almost_equal(s, cs, decimal=4)


class TestMultilevel(unittest.TestCase):
    def test_level(self):
        lvl = Level([3, 10, 11, 11], 1)
        # self.assertEqual(lvl.dim, [1,10,10])
        np.testing.assert_array_equal(lvl.dim, np.array([3, 10, 11, 11]))
        np.testing.assert_array_equal(lvl.dim3, np.array([10, 11, 11]))
        self.assertEqual(lvl.nvox, 3630)
        np.testing.assert_array_equal(lvl.midgrid, np.array([1.5, 5., 5.5, 5.5]))
        self.assertEqual(lvl.ext.minx, None)
        self.assertEqual(lvl.ext.maxx, None)

    def test_multilevel_init(self):
        fixed = np.zeros([4, 4, 4])
        fixed[1] = np.eye(4)
        moving = np.zeros([1, 4, 4, 4])
        u = np.zeros([1, 4, 4, 4])
        h = np.array([1., 1., 1., 1.])

        multi = Multilevel(CYCLE_V2, moving.shape, h, 0.5)
        np.testing.assert_array_equal(multi.level[2].h, np.array([1, 4, 4, 4]))

    def test_translate_image(self):
        u = np.array(np.arange(16)).reshape([1, 1, 4, 4])
        origu = np.array(u)
        (dslice, drow, dcolumn) = (0, 2, 2)
        tr = translate_image(u, dslice, drow, dcolumn)
        ctr = np.array([[[[10, 11, 11, 11], [14, 15, 15, 15], [14, 15, 15, 15], [14, 15, 15, 15]]]])
        np.testing.assert_array_equal(tr, ctr)
        np.testing.assert_array_equal(origu, u)  # Input should not be modified

        u = np.array(np.arange(16)).reshape([1, 4, 4])
        origu = np.array(u)
        (dslice, drow, dcolumn) = (0, 2, 2)
        tr = translate_image(u, dslice, drow, dcolumn)
        ctr = np.array([[[10, 11, 11, 11], [14, 15, 15, 15], [14, 15, 15, 15], [14, 15, 15, 15]]])
        np.testing.assert_array_equal(tr, ctr)
        np.testing.assert_array_equal(origu, u)  # Input should not be modified

        u = np.array(np.arange(16)).reshape([1, 4, 4])
        (dtag, dslice, drow, dcolumn) = (0, 0, 2, 2)
        with self.assertRaises(ValueError):
            tr = translate_image(u, dslice, drow, dcolumn, dtag)

    def test_normgrad(self):
        f = np.zeros([1, 4, 4, 4])
        f[0, :] = np.eye(4)
        origf = np.array(f)
        eta = 0.03
        h = [1., 1., 1.]
        dfn, df, absdfreq = normgrad(f, eta, h)
        np.testing.assert_array_equal(origf, f)
        cdfn0 = np.zeros([1, 4, 4, 4])
        print("test_normgrad: dfn[0]");
        pprint.pprint(dfn[0])
        np.testing.assert_array_equal(dfn[0], cdfn0)
        cdfn1 = np.array([[[[-0.7065, 0.7065, 0, 0], [-0.7065, 0, 0.7065, 0], [0, -0.7065, 0, 0.7065],
                            [0, 0, -0.7065, 0.7065]],
                           [[-0.7065, 0.7065, 0, 0], [-0.7065, 0, 0.7065, 0], [0, -0.7065, 0, 0.7065],
                            [0, 0, -0.7065, 0.7065]],
                           [[-0.7065, 0.7065, 0, 0], [-0.7065, 0, 0.7065, 0], [0, -0.7065, 0, 0.7065],
                            [0, 0, -0.7065, 0.7065]],
                           [[-0.7065, 0.7065, 0, 0], [-0.7065, 0, 0.7065, 0], [0, -0.7065, 0, 0.7065],
                            [0, 0, -0.7065, 0.7065]]]])
        np.testing.assert_array_almost_equal(dfn[1], cdfn1, decimal=4)
        cdfn2 = np.array([[[[-0.7065, -0.7065, 0, 0], [0.7065, 0, -0.7065, 0], [0, 0.7065, 0, -0.7065],
                            [0, 0, 0.7065, 0.7065]],
                           [[-0.7065, -0.7065, 0, 0], [0.7065, 0, -0.7065, 0], [0, 0.7065, 0, -0.7065],
                            [0, 0, 0.7065, 0.7065]],
                           [[-0.7065, -0.7065, 0, 0], [0.7065, 0, -0.7065, 0], [0, 0.7065, 0, -0.7065],
                            [0, 0, 0.7065, 0.7065]],
                           [[-0.7065, -0.7065, 0, 0], [0.7065, 0, -0.7065, 0], [0, 0.7065, 0, -0.7065],
                            [0, 0, 0.7065, 0.7065]]]])
        np.testing.assert_array_almost_equal(dfn[2], cdfn2, decimal=4)

    def test_multilevel(self):
        fixed = np.zeros([8, 8, 8])
        fixed[1] = np.eye(8)
        moving = np.zeros([1, 8, 8, 8])
        moving[0, 2] = np.eye(8)
        u = {}
        u[0] = np.zeros([1, 8, 8, 8])
        u[1] = np.zeros([1, 8, 8, 8])
        u[2] = np.zeros([1, 8, 8, 8])
        u[3] = np.zeros([1, 8, 8, 8])
        h = np.array([1., 1., 1., 1.])
        nudim = 3
        eta = 0.03

        multi = Multilevel(CYCLE_V2, moving.shape, h, 0.5)
        np.testing.assert_array_equal(multi.level[2].h, np.array([1., 4., 4., 4.]))

        multi.set_fixed_image(fixed)
        cfix = np.array([[[0., 0.], [0., 0.]], [[1., 0.], [0., 1.]], [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]])
        np.testing.assert_array_equal(multi.level[2].fixed, cfix)

        multi.set_moving_image(moving)
        """
        for m in range(multi.nmultilevel):
            pprint.pprint(m)
            pprint.pprint(multi.level[m].moving)
            pprint.pprint(multi.level[m].fu)
        """
        cmov = np.array([[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]], [[1., 0.], [0., 1.]], [[0., 0.], [0., 0.]],
                         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]])
        cu = np.array([[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]], [[1., 0.], [0., 1.]], [[0., 0.], [0., 0.]],
                       [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]])
        np.testing.assert_array_equal(multi.level[2].moving, cmov)
        np.testing.assert_array_equal(multi.level[2].fu, cu)

        if isinstance(moving.shape, tuple):
            ndim = len(moving.shape)
        else:
            ndim = moving.shape.size()
        multi.set_deformation_field(None, ndim)
        np.testing.assert_array_equal(multi.level[0].u[0], np.zeros([1, 8, 8, 8]))
        multi.set_deformation_field(u, ndim)
        np.testing.assert_array_equal(multi.level[0].u[0], np.zeros([1, 8, 8, 8]))
        multi.set_gradients(nudim, eta)
        cabsgradg = np.zeros([8, 8, 8])
        cabsgradg[0] = 0.5 * np.eye(8)
        cabsgradg[1] = np.array([ \
            [0.7071, 0.7071, 0, 0, 0, 0, 0, 0], \
            [0.7071, 0, 0.7071, 0, 0, 0, 0, 0], \
            [0, 0.7071, 0, 0.7071, 0, 0, 0, 0], \
            [0, 0, 0.7071, 0, 0.7071, 0, 0, 0], \
            [0, 0, 0, 0.7071, 0, 0.7071, 0, 0], \
            [0, 0, 0, 0, 0.7071, 0, 0.7071, 0], \
            [0, 0, 0, 0, 0, 0.7071, 0, 0.7071], \
            [0, 0, 0, 0, 0, 0, 0.7071, 0.7071]])
        cabsgradg[2] = 0.5 * np.eye(8)
        np.testing.assert_array_almost_equal(multi.level[0].absgradg, cabsgradg, decimal=3)

    def test_solve_nonlinearfp(self):
        print("\ntest_solve_nonlinearfp:\n\n")
        # dim = np.array([4, 4, 4], dtype=int)
        h = np.array([1., 1., 1.])
        # ext = LevelExt()
        # x, ext.minx, ext.maxx = centergrid(dim, h)
        """
        transform = TransformLinear(ext)
        fixed = create_linear_3d_matrix([4,4,4])
        moving = transform.apply(fixed,4,u,x)
        """

        fixed = np.zeros([4, 4, 4])
        fixed[1, :, :] = np.eye(4)
        moving = np.zeros([4, 4, 4])
        moving[1, 0:-1, 1:] = np.eye(3)

        u = {}
        u[0] = np.ones([4, 4, 4])
        u[1] = np.zeros([4, 4, 4])
        u[2] = np.zeros([4, 4, 4])

        nudim = 3
        eta = 0.03

        npreg = NPreg(fixed, prm={'maxniter': 5})
        npreg.cycle = CYCLE_NONE
        # npreg.multi = Multilevel(CYCLE_V2, moving.shape, h, 0.5)
        npreg.multi = Multilevel(CYCLE_NONE, moving.shape, h, 0.5)
        npreg.multi.set_fixed_image(fixed)
        npreg.multi.set_moving_image(moving)
        if isinstance(moving.shape, tuple):
            ndim = len(moving.shape)
        else:
            ndim = moving.shape.size()
        # npreg.multi.set_deformation_field(None, ndim)
        npreg.multi.set_deformation_field(u, ndim)
        npreg.multi.set_gradients(nudim, eta)
        npreg.nudim = 3
        npreg.ndim = 3
        npreg.multigrid = npreg.multi.multigrid
        # npreg.multi.level[0] = npreg.solve_nonlinearfp()
        npreg.solve_nonlinearfp()

        cu0 = np.array([[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]])
        cu1 = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])

        np.testing.assert_array_equal(cu0, u[0])
        np.testing.assert_array_equal(cu1, u[1])
        np.testing.assert_array_equal(cu1, u[2])
        # np.testing.assert_array_almost_equal(moving, fixed, decimal=3)

    def test_register_volume(self):
        print("\ntest_register_volume:\n\n")
        # dim = np.array([4, 4, 4], dtype=int)
        # h = np.array([1., 1., 1.])
        # ext = LevelExt()
        # x, ext.minx, ext.maxx = centergrid(dim, h)

        fixed = np.zeros([4, 4, 4])
        fixed[1, :, :] = np.eye(4)
        fixed = Series(fixed)
        moving = np.zeros([4, 4, 4])
        moving[1, 0:-1, 1:] = np.eye(3)
        moving = Series(moving)

        npreg = NPreg(fixed, prm={'maxniter': 30})
        npreg.cycle = CYCLE_NONE
        print("test_register_volume: moving", type(moving), moving.dtype, moving.shape)
        out = npreg.register_volume(moving)
        # cProfile.run('npreg.register_volume(moving)')

        cu_out = np.array([[[5.94428754e-03, 4.01915792e-03, 4.01422009e-05, 1.00000000e-06],
                            [3.50048338e-03, 4.82599426e-03, 1.64289205e-03, 4.01422009e-05],
                            [0.00000000e+00, 3.50381656e-03, 4.82599426e-03, 4.01915792e-03],
                            [0.00000000e+00, 0.00000000e+00, 3.50048338e-03, 5.94428754e-03]],
                           [[5.95922528e-01, 3.89867840e-01, 3.95280172e-03, 9.91147243e-05],
                            [3.40378483e-01, 4.78189375e-01, 1.62992189e-01, 3.95280172e-03],
                            [0.00000000e+00, 3.46946951e-01, 4.78189375e-01, 3.89867840e-01],
                            [0.00000000e+00, 0.00000000e+00, 3.40378483e-01, 5.95922528e-01]],
                           [[6.06926210e-03, 3.58943918e-03, 3.22997787e-05, 7.99455673e-07],
                            [3.44454750e-03, 4.53255585e-03, 1.44975102e-03, 3.22997787e-05],
                            [0.00000000e+00, 3.15325949e-03, 4.53255585e-03, 3.58943918e-03],
                            [0.00000000e+00, 0.00000000e+00, 3.44454750e-03, 6.06926210e-03]],
                           [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]]
                          )
        np.testing.assert_array_almost_equal(cu_out, out, decimal=2)

    def test_register_series_NONE(self):
        a = Series('data/time.zip', 'time')
        out = register_npreg(0, a, cycle=CYCLE_NONE, prm={'maxniter': 5})

    def test_register_series_V1(self):
        a = Series('data/time.zip', 'time')
        out = register_npreg(0, a, cycle=CYCLE_V1, prm={'maxniter': 5})

    def test_register_series_V2(self):
        a = Series('data/time.zip', 'time')
        out = register_npreg(0, a, cycle=CYCLE_V2, prm={'maxniter': 5})

    def test_register_series_V3(self):
        a = Series('data/time.zip', 'time')
        out = register_npreg(0, a, cycle=CYCLE_V3, prm={'maxniter': 5})

    def test_register_series_W2(self):
        a = Series('data/time.zip', 'time')
        out = register_npreg(0, a, cycle=CYCLE_W2, prm={'maxniter': 5})

    def test_register_series_W3(self):
        a = Series('data/time.zip', 'time')
        out = register_npreg(0, a, cycle=CYCLE_W3, prm={'maxniter': 5})


if __name__ == '__main__':
    unittest.main()
