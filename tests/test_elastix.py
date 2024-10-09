#!/usr/bin/env python3

import unittest
import numpy as np
import pprint
from imagedata.series import Series
import itk

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

    def test_register_elastix_defaultparameters(self):
        a = Series("data/time.zip", "time")
        a2 = np.zeros((a.shape[0], 2*a.shape[1], a.shape[2], a.shape[3]))
        a2[:, 0:3] = a[:]
        a2[:, 3:6] = a[:]
        a = Series(a2, "time")
        parametermap = itk.ParameterObject.New()
        default_translation_map = parametermap.GetDefaultParameterMap('translation')
        parametermap.AddParameterMap(default_translation_map)
        out = register_elastix_parametermap(0, a, parametermap)


    def test_register_elastix_read_parameters(self):
        a = Series("data/time.zip", "time")
        a2 = np.zeros((a.shape[0], 2 * a.shape[1], a.shape[2], a.shape[3]))
        a2[:, 0:3] = a[:]
        a2[:, 3:6] = a[:]
        a = Series(a2, "time")
        parametermap = itk.ParameterObject.New()
        parametermap.AddParameterFile("data/Elastix/Parameters_Rigid.txt")
        out = register_elastix_parametermap(0, a, parametermap)

    def test_register_elastix_series(self):
        a = Series("data/time.zip", "time")
        a2 = np.zeros((a.shape[0], 2 * a.shape[1], a.shape[2], a.shape[3]))
        a2[:, 0:3] = a[:]
        a2[:, 3:6] = a[:]
        fixedSeries = Series(a2[0])
        movingSeries = Series(a2[1])
        fixedImage = itk.GetImageFromArray(np.array(fixedSeries, dtype=float))
        fixedImage.SetSpacing(fixedSeries.spacing.astype(float))
        movingImage = itk.GetImageFromArray(np.array(movingSeries, dtype=float))
        movingImage.SetSpacing(movingSeries.spacing.astype(float))
        parameterMap = itk.ParameterObject.New()
        parameterMap.AddParameterFile("data/Elastix/Parameters_Rigid.txt")

        elastixImageFilter = itk.ElastixRegistrationMethod.New(fixedImage, movingImage)
        elastixImageFilter.SetParameterObject(parameterMap)
        elastixImageFilter.Update()

        resultImage = elastixImageFilter.GetOutput()
        transformParameterMap = elastixImageFilter.GetTransformParameterObject()

        out = itk.GetArrayFromImage(resultImage)
        super_threshold_indices = out > 65500
        out[super_threshold_indices] = 0

        resultSeries = Series(out,
                              template=movingSeries,
                              geometry=fixedSeries)
        pass


if __name__ == '__main__':
    unittest.main()
