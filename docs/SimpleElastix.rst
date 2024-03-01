.. _SimpleElastix:

SimpleElastix Examples
======================

Using a parameter map
---------------------
The following example shows how to register a `Series` using
an `Elastix ParameterMap`.
See
https://simpleelastix.readthedocs.io/ParameterMaps.html
for details on how to set up a ParameterMap in `SimpleElastix`.

.. code-block:: python

    from imagedata import Series
    from imagedata_registration.Elastix import register_elastix_parametermap
    import SimpleITK as sitk

    # fixed can be either a Series volume,
    # or an index (int) into moving Series
    # moving can be a 3D or 4D Series instance
    moving = Series("data/", "time")
    fixed = 10
    parametermap = sitk.GetDefaultParameterMap("translation")
    out = register_elastix_parametermap(fixed, moving, parametermap)


Using SimpleElastix's Object-Oriented Interface
-----------------------------------------------
For complete control the SimpleElastix's object-orient interface can be used directly.
The code here converts Series objects to SimpleElastix Image objects, then uses the SimpleElastix
methods on these objects, and converts the final resultImage to Series again.
This way all SimpleElastix methods are available.

.. code-block:: python

    from imagedata import Series
    import SimpleITK as sitk

    fixedSeries = Series('fixed')
    movingSeries = Series('moving')
    fixedImage = sitk.GetImageFromArray(np.array(fixedSeries, dtype=float))
    fixedImage.SetSpacing(fixedSeries.spacing.astype(float))
    movingImage = sitk.GetImageFromArray(np.array(movingSeries, dtype=float))
    movingImage.SetSpacing(movingSeries.spacing.astype(float))
    parameterMap = sitk.GetDefaultParameterMap('translation')
    # parameterMap = sitk.ReadParameterFile("Parameters_Rigid.txt")

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.Execute()

    resultImage = elastixImageFilter.GetResultImage()
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    out = sitk.GetArrayFromImage(resultImage)
    super_threshold_indices = out > 65500
    out[super_threshold_indices] = 0

    resultSeries = Series(out,
                          template=movingSeries,
                          geometry=fixedSeries)
    resultSeries.write('result', formats=['dicom'])


Using SimpleElastix's Object-Oriented Interface (time-dependent Series)
-----------------------------------------------------------------------
This example builds on the previous one, adding the code to register a time Series,
time-point by time-point.

.. code-block:: python

    from imagedata import Series
    import SimpleITK as sitk

    fixedSeries = Series('fixed')
    movingSeries = Series('moving', 'time')
    fixedImage = sitk.GetImageFromArray(np.array(fixedSeries, dtype=float))
    fixedImage.SetSpacing(fixedSeries.spacing.astype(float))

    parameterMap = sitk.GetDefaultParameterMap('translation')
    # parameterMap = sitk.ReadParameterFile("Parameters_Rigid.txt")

    shape = (movingSeries.shape[0],) + fixedSeries.shape
    tags = movingSeries.tags[0]

    out = np.zeros(shape, dtype=movingSeries.dtype)
    transformParameterMap = []

    for t, tag in enumerate(tags):
        movingImage = sitk.GetImageFromArray(np.array(movingSeries[t], dtype=float))
        movingImage.SetSpacing(movingSeries.spacing.astype(float))

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixedImage)
        elastixImageFilter.SetMovingImage(movingImage)
        elastixImageFilter.SetParameterMap(parameterMap)
        elastixImageFilter.Execute()

        resultImage = elastixImageFilter.GetResultImage()
        transformParameterMap.append(elastixImageFilter.GetTransformParameterMap())

        out[t] = sitk.GetArrayFromImage(resultImage)
    super_threshold_indices = out > 65500
    out[super_threshold_indices] = 0

    resultSeries = Series(out,
                          input_order=movingSeries.input_order,
                          template=movingSeries,
                          geometry=fixedSeries)
    resultSeries.tags = moving.tags
    resultSeries.axes[0] = movingSeries.axes[0]
    resultSeries.write('result', formats=['dicom'])


A skeleton
----------

A function `register_elastix` is provided here.
This function will register a **moving** `Series` to a **fixed** `Series`.
`register_elastix` is based on one of the `SimpleElastix` examples
in
https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod1_docs.html
and can serve as an example for using `ITK/Elastix` methods.


.. code-block:: python

    from imagedata import Series
    from imagedata_registration.Elastix import register_elastix

    # fixed can be either a Series volume,
    # or an index (int) into moving Series
    # moving can be a 3D or 4D Series instance
    moving = Series("data/", "time")
    fixed = 10
    out = register_elastix(fixed, moving)


Documentation on ITK / Elastix
------------------------------
* SimpleElastix: https://simpleelastix.readthedocs.io/
* SimpleITK: https://simpleitk.readthedocs.io/
