.. _ITKElastix:

ITK-Elastix Examples
======================

Using a parameter map
---------------------
The following example shows how to register a `Series` using
an `Elastix ParameterObject`.
See
https://github.com/InsightSoftwareConsortium/ITKElastix/tree/main/examples
for details on how to set up a ParameterObject in `ITK-Elastix`.

.. code-block:: python

    from imagedata import Series
    from imagedata_registration.Elastix import register_elastix_parametermap
    import itk

    # fixed can be either a Series volume,
    # or an index (int) into moving Series
    # moving can be a 3D or 4D Series instance
    moving = Series("data/", "time")
    fixed = 10
    parametermap = itk.ParameterObject.New()
    default_rigid_parameter_map = parametermap.GetDefaultParameterMap('rigid')
    parametermap.AddParameterMap(default_rigid_parameter_map)
    out = register_elastix_parametermap(fixed, moving, parametermap)


Using ITK-Elastix's Object-Oriented Interface
-----------------------------------------------
For complete control the ITK-Elastix's object-orient interface can be used directly.
The code here converts Series objects to ITK-Elastix Image objects, then uses the ITK-Elastix
methods on these objects, and converts the final resultImage to Series again.
This way all ITK-Elastix methods are available.

.. code-block:: python

    from imagedata import Series
    import itk

    fixedSeries = Series('fixed')
    movingSeries = Series('moving')
    fixedImage = itk.GetImageFromArray(np.array(fixedSeries, dtype=float))
    fixedImage.SetSpacing(fixedSeries.spacing.astype(float))
    movingImage = itk.GetImageFromArray(np.array(movingSeries, dtype=float))
    movingImage.SetSpacing(movingSeries.spacing.astype(float))

    parameterMap = itk.ParameterObject.New()
    default_rigid_parameter_map = parameterMap.GetDefaultParameterMap('rigid')
    parameterMap.AddParameterMap(default_rigid_parameter_map)

    elastixImageFilter = itk.ElastixRegistrationMethod.New(fixed_itk, moving_itk)
    elastixImageFilter.SetParameterObject(parametermap)
    elastixImageFilter.UpdateLargestPossibleRegion()

    resultImage = elastixImageFilter.GetOutput()
    transformParameterMap = elastixImageFilter.GetTransformParameterObject()

    out = itk.GetArrayFromImage(resultImage)
    super_threshold_indices = out > 65500
    out[super_threshold_indices] = 0

    resultSeries = Series(out,
                          template=movingSeries,
                          geometry=fixedSeries)
    resultSeries.write('result', formats=['dicom'])


Using ITL-Elastix's Object-Oriented Interface (time-dependent Series)
-----------------------------------------------------------------------
This example builds on the previous one, adding the code to register a time Series,
time-point by time-point.

.. code-block:: python

    from imagedata import Series
    import itk

    fixedSeries = Series('fixed')
    movingSeries = Series('moving', 'time')
    fixedImage = itk.GetImageFromArray(np.array(fixedSeries, dtype=float))
    fixedImage.SetSpacing(fixedSeries.spacing.astype(float))

    parameterMap = itk.ParameterObject.New()
    default_rigid_parameter_map = parameterMap.GetDefaultParameterMap('rigid')
    parameterMap.AddParameterMap(default_rigid_parameter_map)

    shape = (movingSeries.shape[0],) + fixedSeries.shape
    tags = movingSeries.tags[0]

    out = np.zeros(shape, dtype=movingSeries.dtype)
    transformParameterMap = []

    for t, tag in enumerate(tags):
        movingImage = itk.GetImageFromArray(np.array(movingSeries[t], dtype=float))
        movingImage.SetSpacing(movingSeries.spacing.astype(float))

        elastixImageFilter = itk.ElastixRegistrationMethod.New(fixed_itk, moving_itk)
        elastixImageFilter.SetParameterObject(parameterMap)
        elastixImageFilter.UpdateLargestPossibleRegion()
        resultImage = elastixImageFilter.GetOutput()
        transformParameter.append(elastixImageFilter.GetTransformParameterObject())

        out[t] = itk.GetArrayFromImage(resultImage)
    super_threshold_indices = out > 65500
    out[super_threshold_indices] = 0

    resultSeries = Series(out,
                          input_order=movingSeries.input_order,
                          template=movingSeries,
                          geometry=fixedSeries)
    resultSeries.tags = moving.tags
    resultSeries.axes[0] = movingSeries.axes[0]
    resultSeries.write('result', formats=['dicom'])


Documentation on ITK-Elastix
------------------------------
* ITK-Elastix: https://github.com/InsightSoftwareConsortium/ITKElastix
