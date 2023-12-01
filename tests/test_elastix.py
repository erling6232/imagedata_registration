#!/usr/bin/env python3

import unittest
import numpy as np
import pprint
from imagedata.series import Series
import SimpleITK as sitk

from src.imagedata_registration.Elastix import register_elastix, register_elastix_parametermap


class TestElastixRegistration(unittest.TestCase):
    def test_register_elastix(self):
        a = Series("data/time.zip", "time")
        a2 = np.zeros((a.shape[0], 2*a.shape[1], a.shape[2], a.shape[3]))
        a2[:, 0:3] = a[:]
        a2[:, 3:6] = a[:]
        a = Series(a2, "time")
        # a.seriesDescription="Stacked"
        out = register_elastix(0, a, options={"cost": "corratio"})

    def test_register_elastix(self):
        a = Series("data/time.zip", "time")
        a2 = np.zeros((a.shape[0], 2*a.shape[1], a.shape[2], a.shape[3]))
        a2[:, 0:3] = a[:]
        a2[:, 3:6] = a[:]
        a = Series(a2, "time")
        parametermap = sitk.GetDefaultParameterMap("translation")
        out = register_elastix_parametermap(0, a, parametermap)


if __name__ == '__main__':
    unittest.main()
