#!/usr/bin/env python3

import unittest
import numpy as np
import pprint
from imagedata.series import Series

from src.imagedata_registration.FSL import register_fsl


class TestFSLRegistration(unittest.TestCase):
    def test_register_fsl(self):
        a = Series('data/time.zip', 'time')
        out = register_fsl(0, a, options={"cost": "corratio"})
        print(out.shape)


if __name__ == '__main__':
    unittest.main()
