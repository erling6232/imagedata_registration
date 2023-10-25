"""
FSL based image registration.
This is the where FSL registration is called.
"""

from typing import Callable, ClassVar, Dict, Union
import nipype.interfaces.fsl as fsl
import tempfile
from pathlib import Path
from imagedata import Series


def register_fsl(
        fixed: Union[int, Series],
        moving: Series,
        method: Callable[[], fsl.FSLCommand] = fsl.MCFLIRT,
        options: Dict = {}) -> Series:
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

    if moving.ndim > fixed_volume.ndim:
        shape = (moving.shape[0],) + fixed_volume.shape
        tags = moving.tags[0]
    else:
        shape = fixed_volume.shape
        tags = [None]

    with tempfile.TemporaryDirectory() as tmp:
        print('\nPreparing for MCFLIRT ...')
        p = Path(tmp)
        fixed_path = None
        if not issubclass(type(fixed), int):
            tmp_fixed = p / 'fixed'
            fixed.write(tmp_fixed, formats=['nifti'])
            fixed_path = list(tmp_fixed.glob('*'))[0]
        tmp_moving = p / 'moving'
        moving.write(tmp_moving, formats=['nifti'])
        moving_path = list(tmp_moving.glob('*'))[0]

        print('MCFLIRT running ...')
        tmp_out = p / 'out.nii.gz'

        reg_method = method()
        reg_method.inputs.in_file = str(moving_path)
        if fixed_path is None:
            reg_method.inputs.ref_vol = fixed
        else:
            reg_method.inputs.ref_file = str(fixed_path)
        reg_method.inputs.out_file = str(tmp_out)
        for key in options.keys():
            print("{} -> {}".format(key, options[key]))
            setattr(reg_method.inputs, key, options[key])
        # mcflt.inputs.cost = "corratio"
        # mcflt.inputs.cost     = "normcorr"
        print('{}'.format(reg_method.cmdline))
        result = reg_method.run()

        out = Series(tmp_out, input_order=moving.input_order, template=moving, geometry=fixed_volume)
        out.tags = moving.tags
        # out.axes = moving.axes
        out.seriesDescription = 'MCFLIRT {}'.format(reg_method.inputs.cost)
        super_threshold_indices = out > 65500
        out[super_threshold_indices] = 0

        print('MCFLIRT ended.\n')
        return out

