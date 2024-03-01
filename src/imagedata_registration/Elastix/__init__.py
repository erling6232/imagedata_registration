from typing import Dict, Union
import numpy as np
from imagedata.series import Series
import SimpleITK as sitk


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
    fixed_itk = sitk.GetImageFromArray(np.array(fixed_volume, dtype=float))
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
            moving_itk = sitk.GetImageFromArray(np.array(moving, dtype=float))
        else:
            moving_itk = sitk.GetImageFromArray(np.array(moving[t], dtype=float))
        moving_itk.SetSpacing(moving.spacing.astype(float))

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMeanSquares()
        R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 200)
        R.SetInitialTransform(sitk.TranslationTransform(fixed_itk.GetDimension()))
        R.SetInterpolator(sitk.sitkLinear)
        outTx = R.Execute(fixed_itk, moving_itk)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_itk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(outTx)
        out_itk = resampler.Execute(moving_itk)

        if tag is None:
            # out = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
            out = sitk.GetArrayFromImage(out_itk)
        else:
            # out[t] = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
            out[t] = sitk.GetArrayFromImage(out_itk)
    print('------DONE---------------------------------------')

    super_threshold_indices = out > 65500
    out[super_threshold_indices] = 0

    res = Series(out, input_order=moving.input_order, template=moving, geometry=fixed_volume)
    if res.ndim > fixed_volume.ndim:
        res.tags = moving.tags
        res.axes[0] = moving.axes[0]
    try:
        res.seriesDescription += ' ITK Elastix'
    except ValueError:
        res.seriesDescription = 'ITK Elastix'
    return res


def register_elastix_parametermap(
        fixed: Union[int, Series],
        moving: Series,
        parametermap: sitk.ParameterMap) -> Series:
    """Register a series using ITK Elastix methods.

    Args:
        fixed (int or Series): Fixed volume, or index into moving
        moving (Series): Moving volume(s)
        parametermap (SimpleITK.ParameterMap): Elastix ParameterMap
    Returns:
        Registered series (Series)
    """

    if issubclass(type(fixed), int):
        fixed_volume = moving[fixed]
    else:
        fixed_volume = fixed
    fixed_itk = sitk.GetImageFromArray(np.array(fixed_volume, dtype=float))
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
            moving_itk = sitk.GetImageFromArray(np.array(moving, dtype=float))
        else:
            moving_itk = sitk.GetImageFromArray(np.array(moving[t], dtype=float))
        moving_itk.SetSpacing(moving.spacing.astype(float))

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetParameterMap(parametermap)
        elastixImageFilter.SetFixedImage(fixed_itk)
        elastixImageFilter.SetMovingImage(moving_itk)
        elastixImageFilter.Execute()
        out_itk = elastixImageFilter.GetResultImage()
        transform = elastixImageFilter.GetTransformParameterMap()

        if tag is None:
            # out = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
            out = sitk.GetArrayFromImage(out_itk)
        else:
            # out[t] = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
            out[t] = sitk.GetArrayFromImage(out_itk)
    print('------DONE---------------------------------------')

    super_threshold_indices = out > 65500
    out[super_threshold_indices] = 0

    res = Series(out, input_order=moving.input_order, template=moving, geometry=fixed_volume)
    if res.ndim > fixed_volume.ndim:
        res.tags = moving.tags
        res.axes[0] = moving.axes[0]
    try:
        res.seriesDescription += ' ITK Elastix'
    except ValueError:
        res.seriesDescription = 'ITK Elastix'
    return res
