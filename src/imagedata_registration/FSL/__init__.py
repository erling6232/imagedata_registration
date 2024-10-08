"""
FSL based image registration.
This is the where FSL registration is called.
"""

from typing import Callable, Dict, Union
import nipype.interfaces.fsl as fsl
import tempfile
from pathlib import Path
from imagedata import Series


def register_fsl(
        fixed: Union[int, Series],
        moving: Series,
        method: Callable[[], fsl.FSLCommand] = fsl.MCFLIRT,
        options: Dict = {'cost': 'corratio'}) -> Series:
    """Register a series using FSL methods.

    Args:
        fixed (int or Series): Fixed volume, or index into moving
        moving (Series): Moving volume(s)
        method (int): FSL method. Default: MCFLIRT
        options (dict): Options to method
    Returns:
        Registered series (Series)
    """

    if issubclass(type(fixed), int):
        fixed_volume = moving[fixed]
    else:
        fixed_volume = fixed
    # cost = None if "cost" not in options else options["cost"]

    # if moving.ndim > fixed_volume.ndim:
    #     shape = (moving.shape[0],) + fixed_volume.shape
    #     tags = moving.tags[0]
    # else:
    #     shape = fixed_volume.shape
    #     tags = [None]

    with tempfile.TemporaryDirectory() as tmp:
        print('\nPreparing for FSL ...')
        p = Path(tmp)
        fixed_path = None
        if not issubclass(type(fixed), int):
            tmp_fixed = p / 'fixed.nii.gz'
            fixed.write(tmp_fixed, formats=['nifti'])
        tmp_moving = p / 'moving.nii.gz'
        moving.write(tmp_moving, formats=['nifti'])

        print('FSL running ...')
        tmp_out = p / 'out.nii.gz'

        reg_method = method()
        reg_method.inputs.in_file = str(tmp_moving)
        if fixed_path is None:
            reg_method.inputs.ref_vol = fixed
        else:
            reg_method.inputs.ref_file = str(tmp_fixed)
        reg_method.inputs.out_file = str(tmp_out)
        for key in options.keys():
            print("{} -> {}".format(key, options[key]))
            setattr(reg_method.inputs, key, options[key])
        # mcflt.inputs.cost = "corratio"
        # mcflt.inputs.cost     = "normcorr"
        print('{}'.format(reg_method.cmdline))
        _ = reg_method.run()

        out = Series(tmp_out, input_order=moving.input_order, template=moving, geometry=fixed_volume)
        out.tags = moving.tags
        # out.axes = moving.axes
        super_threshold_indices = out > 65500
        out[super_threshold_indices] = 0
        if out.ndim > fixed_volume.ndim:
            out.tags = moving.tags
            out.axes = out.axes._replace(**{out.input_order: moving.axes[0]})
        try:
            out.seriesDescription += ' {} {}'.format(
                reg_method.cmd,
                reg_method.inputs.cost)
        except ValueError:
            out.seriesDescription = '{} {}'.format(
                reg_method.cmd,
                reg_method.inputs.cost)

        print('FSL ended.\n')
        return out

