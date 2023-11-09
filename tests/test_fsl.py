#!/usr/bin/env python3

import os
import unittest
import numpy as np
import pprint
import tempfile
from imagedata.series import Series

from src.imagedata_registration.FSL import register_fsl


class TestFSLRegistration(unittest.TestCase):
    def test_register_fsl(self):
        if os.getenv("GITHUB_ACTION") is not None:
            return
        a = Series('data/time.zip', 'time')
        out = register_fsl(0, a, options={"cost": "corratio"})
        np.testing.assert_array_equal(out.tags[0], a.tags[0])
        self.assertEqual(out.axes[0], a.axes[0])
        with tempfile.TemporaryDirectory() as d:
            out.write(d, formats=['dicom'])


if __name__ == '__main__':
    unittest.main()
