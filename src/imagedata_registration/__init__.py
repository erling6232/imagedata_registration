from typing import Callable, Dict, Union
import numpy as np
from imagedata.series import Series


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

    import SimpleITK as sitk

    if issubclass(type(fixed), int):
        fixed_volume = moving[fixed]
    else:
        fixed_volume = fixed
    fixed_itk = sitk.GetImageFromArray(fixed_volume)
    sp = fixed_volume.spacing
    fixed_itk.SetSpacing([sp[2], sp[1], sp[0]])

    if moving.ndim > fixed_volume.ndim:
        shape = (moving.shape[0],) + fixed_volume.shape
        tags = moving.tags[0]
    else:
        shape = fixed_volume.shape
        tags = [None]

    #if si_fixed_mask is not None:
    #    fixed_mask = sitk.GetImageFromArray(si_fixed_mask)
    #    fixed_mask.SetSpacing([sp[2], sp[1], sp[0]])

    out = np.zeros(shape, dtype=moving.dtype)
    for t, tag in enumerate(tags):
        print('-------------------------------------------------')
        print('Elastix register {} of {}'.format(t + 1, len(tags)))
        if tag is None:
            moving_itk = sitk.GetImageFromArray(moving)
        else:
            moving_itk = sitk.GetImageFromArray(moving[t])
        sp = moving.spacing
        moving_itk.SetSpacing([sp[-1], sp[-2], sp[-3]])

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed_itk)
        elastixImageFilter.SetMovingImage(moving_itk)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
#        if si_fixed_mask is not None:
#            elastixImageFilter.SetFixedMask(fixed_mask_itk)
        elastixImageFilter.Execute()
        if tag is None:
            out = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
        else:
            out[t] = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
        print('------DONE---------------------------------------')

    super_threshold_indices = out > 65500
    out[super_threshold_indices] = 0

    res = Series(out, input_order=moving.input_order, template=moving, geometry=fixed_volume)
    res.tags = moving.tags
    # res.axes = moving.axes
    res.seriesDescription = 'ITK Elastix {}'.format(0)
    return res
