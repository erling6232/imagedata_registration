from typing import Dict, Union
import numpy as np
from imagedata.series import Series
import itk


def register_elastix(
        fixed: Union[int, Series],
        moving: Series,
        options: Dict = {}) -> Series:
    """Register a series using ITK Elastix methods.

    Args:
        fixed (int or Series): Fixed volume, or index into moving
        moving (Series): Moving volume(s)
        options (dict): Options to method
    Returns:
        Registered series (Series)
    """

    if issubclass(type(fixed), int):
        fixed_volume = moving[fixed]
    else:
        fixed_volume = fixed
    fixed_itk = itk.GetImageFromArray(np.array(fixed_volume, dtype=float))
    fixed_itk.SetSpacing(fixed_volume.spacing.astype(float))

    if moving.ndim > fixed_volume.ndim:
        shape = (moving.shape[0],) + fixed_volume.shape
        tags = moving.tags[0]
    else:
        shape = fixed_volume.shape
        tags = [None]

    out = np.zeros(shape, dtype=moving.dtype)
    print('-------------------------------------------------')
    for t, tag in enumerate(tags):
        print('Elastix register {} of {}'.format(t + 1, len(tags)))
        if tag is None:
            moving_itk = itk.GetImageFromArray(np.array(moving, dtype=float))
        else:
            moving_itk = itk.GetImageFromArray(np.array(moving[t], dtype=float))
        moving_itk.SetSpacing(moving.spacing.astype(float))

        # R = itk.ImageRegistrationMethod()
        # R.SetMetricAsMeanSquares()
        # R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 200)
        # R.SetInitialTransform(itk.TranslationTransform(fixed_itk.GetDimension()))
        # R.SetInterpolator(itk.sitkLinear)
        # R.Update()
        # # outTx = R.Execute(fixed_itk, moving_itk)
        # outTx = R.GetOutput()
        # # rtp = elastix_obj.GetTransformParameterObject()
        #
        # resampler = itk.ResampleImageFilter()
        # resampler.SetReferenceImage(fixed_itk)
        # resampler.SetInterpolator(itk.sitkLinear)
        # resampler.SetDefaultPixelValue(100)
        # resampler.SetTransform(outTx)
        # out_itk = resampler.Execute(moving_itk)
        parametermap = itk.ParameterObject.New()
        default_rigid_parameter_map = parametermap.GetDefaultParameterMap('rigid')
        parametermap.AddParameterMap(default_rigid_parameter_map)

        elastixImageFilter = itk.ElastixRegistrationMethod.New(fixed_itk, moving_itk)
        elastixImageFilter.SetParameterObject(parametermap)
        elastixImageFilter.UpdateLargestPossibleRegion()
        out_itk = elastixImageFilter.GetOutput()
        transform = elastixImageFilter.GetTransformParameterObject()

        if tag is None:
            # out = itk.GetArrayFromImage(elastixImageFilter.GetResultImage())
            out = itk.GetArrayFromImage(out_itk)
        else:
            # out[t] = itk.GetArrayFromImage(elastixImageFilter.GetResultImage())
            out[t] = itk.GetArrayFromImage(out_itk)
    print('------DONE---------------------------------------')

    super_threshold_indices = out > 65500
    out[super_threshold_indices] = 0

    res = Series(out, input_order=moving.input_order, template=moving, geometry=fixed_volume)
    if res.ndim > fixed_volume.ndim:
        res.tags = moving.tags
        res.axes = res.axes._replace(**{res.input_order: moving.axes[0]})
    try:
        res.seriesDescription += ' ITK Elastix'
    except ValueError:
        res.seriesDescription = 'ITK Elastix'
    return res


def register_elastix_parametermap(
        fixed: Union[int, Series],
        moving: Series,
        parametermap) -> Series:
    """Register a series using ITK Elastix methods.

    Args:
        fixed (int or Series): Fixed volume, or index into moving
        moving (Series): Moving volume(s)
        parametermap (ParameterMap): Elastix ParameterMap
    Returns:
        Registered series (Series)
    """

    if issubclass(type(fixed), int):
        fixed_volume = moving[fixed]
    else:
        fixed_volume = fixed
    fixed_itk = itk.GetImageFromArray(np.array(fixed_volume, dtype=float))
    fixed_itk.SetSpacing(fixed_volume.spacing.astype(float))

    if moving.ndim > fixed_volume.ndim:
        shape = (moving.shape[0],) + fixed_volume.shape
        tags = moving.tags[0]
    else:
        shape = fixed_volume.shape
        tags = [None]

    out = np.zeros(shape, dtype=moving.dtype)
    print('-------------------------------------------------')
    for t, tag in enumerate(tags):
        print('Elastix register {} of {}'.format(t + 1, len(tags)))
        if tag is None:
            moving_itk = itk.GetImageFromArray(np.array(moving, dtype=float))
        else:
            moving_itk = itk.GetImageFromArray(np.array(moving[t], dtype=float))
        moving_itk.SetSpacing(moving.spacing.astype(float))

        elastixImageFilter = itk.ElastixRegistrationMethod.New(fixed_itk, moving_itk)
        elastixImageFilter.SetParameterObject(parametermap)
        elastixImageFilter.UpdateLargestPossibleRegion()
        out_itk = elastixImageFilter.GetOutput()
        transform = elastixImageFilter.GetTransformParameterObject()

        if tag is None:
            out = itk.GetArrayFromImage(out_itk)
        else:
            out[t] = itk.GetArrayFromImage(out_itk)
    print('------DONE---------------------------------------')

    super_threshold_indices = out > 65500
    out[super_threshold_indices] = 0

    res = Series(out, input_order=moving.input_order, template=moving, geometry=fixed_volume)
    if res.ndim > fixed_volume.ndim:
        res.tags = moving.tags
        res.axes = res.axes._replace(**{res.input_order: moving.axes[0]})
    try:
        res.seriesDescription += ' ITK Elastix'
    except ValueError:
        res.seriesDescription = 'ITK Elastix'
    return res
