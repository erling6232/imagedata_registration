#!/usr/bin/env python3

import unittest
import numpy as np
import pprint
from imagedata.series import Series

from src.imagedata_registration import register_elastix


class TestElastixRegistration(unittest.TestCase):
    def test_register_elastix(self):
        a = Series('data/time.zip', 'time')
        out = register_elastix(0, a, options={"cost": "corratio"})


if __name__ == '__main__':
    unittest.main()
