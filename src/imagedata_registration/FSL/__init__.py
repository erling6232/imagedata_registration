"""
FSL based image registration.
This is the where FSL registration is called.
"""

from typing import Callable, Dict, Union
import os.path
import numpy as np
import nipype.interfaces.fsl as fsl
import tempfile
from pathlib import Path
from imagedata import Series
from imagedata.apps.diffusion import write_b_vector_file, write_b_value_file


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


def _create_acq_param(filename: str):
    with open(filename, 'w') as f:
        f.write('0 1 0 0.1\n')
        f.write('0 -1 0 0.1\n')


def topup(tmp: str, ap: Series, pa: Series) -> Series:
    """Apply FSL topup to correct distortions caused by magnetic field inhomogeneities"""

    ap_pa = Series(np.array([ap, pa]), 'time', template=ap, geometry=ap)
    acq_param = os.path.join(tmp, 'topup_encoding.txt')
    _create_acq_param(acq_param)
    main = os.path.join(tmp, 'AP_PA.nii.gz')
    out = os.path.join(tmp, 'AP_PA_topup')
    ap_pa.write(main, formats=['nifti'])

    topup = fsl.TOPUP()
    topup.inputs.in_file = main
    topup.inputs.encoding_file = acq_param
    topup.inputs.output_type = 'NIFTI_GZ'
    topup.inputs.out_base = out
    topup.inputs.out_field = out

    print('FSL TOPUP running ...')
    res = topup.run(cwd=tmp)
    return Series(res.outputs.out_fieldcoef, template=ap)


def apply_topup(tmp: str, img: Series | str,
                fieldcoef: Series | str | None = None,
                movpar: Series | str | None = None
                ) -> Series:

    if issubclass(type(img), Series):
        img_main = os.path.join(tmp, 'apply_img.nii.gz')
        img.write(img_main, formats=['nifti'])
    else:
        img_main = img
        img = Series(img)
    assert os.path.exists(img_main), f"TOPUP input ({img_main}) does not exist"

    acq_param = os.path.join(tmp, 'topup_encoding.txt')
    _create_acq_param(acq_param)
    img_corrected = os.path.join(tmp, 'img_corrected.nii.gz')

    if fieldcoef is None:
        in_fieldcoef = os.path.join(tmp, 'AP_PA_topup_fieldcoef.nii.gz')
    elif issubclass(type(fieldcoef), Series):
        in_fieldcoef = os.path.join(tmp, 'AP_PA_topup_fieldcoef.nii.gz')
        fieldcoef.write(in_fieldcoef, formats=['nifti'])
    else:
        in_fieldcoef = fieldcoef
    assert os.path.exists(in_fieldcoef), f"TOPUP fieldcoef ({in_fieldcoef} does not exist"

    if movpar is None:
        in_movpar = os.path.join(tmp, 'AP_PA_topup_movpar.txt')
    else:
        in_movpar = movpar
    assert os.path.exists(in_movpar), f"TOPUP movpar ({in_movpar} does not exist"

    apply_topup = fsl.ApplyTOPUP()
    apply_topup.inputs.in_files = img_main
    apply_topup.inputs.encoding_file = acq_param
    apply_topup.inputs.in_index = [1]
    apply_topup.inputs.in_topup_fieldcoef = in_fieldcoef
    apply_topup.inputs.in_topup_movpar = in_movpar
    apply_topup.inputs.output_type = 'NIFTI_GZ'
    apply_topup.inputs.out_corrected = img_corrected
    apply_topup.inputs.method = 'jac'

    print('FSL ApplyTOPUP running ...')
    apply_res = apply_topup.run(cwd=tmp)

    return Series(apply_res.outputs.out_corrected, template=img)


def bet(tmp: str, img: Series | str, frac: float = 0.5, mask: bool = False, skull: bool = False)\
        -> dict[str, Series]:
    """FSL BET wrapper for skull stripping

    Args:
        img (Series): input image
        frac (float): fractional intensity threshold (0..1); default=0.5; smaller values give larger brain outline estimates
        mask (bool): whether boolean mask is created
        skull (bool): whether skull is extracted

    Returns:
        dict[str, Series]:
            'out': extracted brain image
            'mask': brain mask (if mask=True)
            'skull': extracted skull (if skull=True)
    """
    if issubclass(type(img), Series):
        in_file = os.path.join(tmp, 'img.nii.gz')
        img.write(in_file, formats=['nifti'])
    else:
        in_file = img

    out_file = os.path.join(tmp, 'bet.nii.gz')

    btr = fsl.BET()
    btr.inputs.in_file = in_file
    btr.inputs.frac = frac
    btr.inputs.out_file = out_file
    btr.inputs.mask = mask
    btr.inputs.skull = skull
    btr.inputs.output_type = 'NIFTI_GZ'

    print('FSL BET running ...')
    res = btr.run(cwd=tmp)

    results = {}
    results['out'] = Series(res.outputs.out_file, dtype=img.dtype, template=img)
    if mask:
        results['mask'] = Series(res.outputs.mask_file, dtype=bool, template=img)
    if skull:
        results['skull'] = Series(res.outputs.skull_file, dtype=img.dtype, template=img)
    return results


def eddy(tmp: str, img: Series | str, mask: Series | str,
         topup: Series | str,
         niter: int=5,
         use_cuda: bool=False,
         repol: bool=False,
         is_shelled: bool=False,
         mporder: int=0,
         flm: str='quadratic',
         fwhm: float | list = 0) -> Series:
    """FSL EDDY correction
    """
    if issubclass(type(img), Series):
        in_file = os.path.join(tmp, 'img.nii.gz')
        img.write(in_file, formats=['nifti'])
    else:
        in_file = img
        img = Series(img)
    if issubclass(type(mask), Series):
        in_mask = os.path.join(tmp, 'mask.nii.gz')
        mask.write(in_mask, formats=['nifti'])
    else:  # str
        in_mask = mask
    if issubclass(type(topup), Series):
        in_topup = os.path.join(tmp, 'topup.nii.gz')
        topup.write(in_topup, formats=['nifti'])
    else:
        in_topup = topup
    out_file = os.path.join(tmp, 'eddy.nii.gz')

    in_index = os.path.join(tmp, 'epi_index.txt')
    with open(in_index, 'w') as in_f:
        for _ in range(img.shape[0]):
            in_f.write('1\n')

    acq_param = os.path.join(tmp, 'acq_param.txt')
    _create_acq_param(acq_param)

    in_bvals = os.path.join(tmp, 'bvals.txt')
    in_bvecs = os.path.join(tmp, 'bvecs.txt')

    bvals = []
    bvecs = []
    for _ in range(img.shape[0]):
        _t = img.tags[0][_]
        bvals.append(_t[0][0])
        bvecs.append(_t[0][1])
    write_b_value_file(in_bvals, bvals)
    write_b_vector_file(in_bvecs, bvecs)

    _eddy = fsl.Eddy()
    _eddy.inputs.in_file = in_file
    _eddy.inputs.in_mask = in_mask
    _eddy.inputs.in_index = in_index
    _eddy.inputs.in_acqp = acq_param
    _eddy.inputs.in_topup_fieldcoef = in_topup + '_fieldcoef.nii.gz'
    _eddy.inputs.in_topup_movpar = in_topup + '_movpar.txt'
    _eddy.inputs.in_bvec = in_bvecs
    _eddy.inputs.in_bval = in_bvals
    _eddy.inputs.flm = flm
    _eddy.inputs.fwhm = fwhm
    _eddy.inputs.use_cuda = use_cuda
    _eddy.inputs.niter = niter
    _eddy.inputs.repol = repol
    _eddy.inputs.mporder = mporder
    _eddy.inputs.is_shelled = is_shelled
    _eddy.inputs.output_type = 'NIFTI_GZ'

    out_file = os.path.join(tmp, 'eddy_unwarped')
    _eddy.inputs.out_base = out_file

    print('FSL EDDY running ...')
    res = _eddy.run(cwd=tmp)
    results = {}
    results['out_corrected'] = Series(res.outputs.out_corrected, dtype=img.dtype, template=img)
    return results
