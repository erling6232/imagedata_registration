.. _FSL:

FSL Examples
============

MCFLIRT registration
--------------------

A function `register_fsl` is provided here.
This function will register a moving Series to a fixed Series.
The default registration method is **fsl.MCFLIRT**.
The function will accept other registration methods.

Using **MCFLIRT** module:

.. code-block:: python

    from imagedata import Series
    from imagedata_registration.FSL import register_fsl
    import nipype.interfaces.fsl as fsl

    # fixed can be either a Series volume,
    # or an index (int) into moving Series
    # moving can be a 3D or 4D Series instance
    moving = Series("data/", "time")
    fixed = 10
    out = register_fsl(
        fixed,
        moving,
        method=fsl.MCFLIRT,
        options={
            'cost': 'corratio'
        }
    )



DTI Eddy correction
-------------------

The following example will use FSL **topup**, **bet** and **eddy** to
correct a DTI dataset.
Two inputs are required, a posterior-anterior phase-encoded direction series,
and a DTI anterior-posterior phase-encoded series.

Notice how these FSL commands are run in a common working directory,
because these commands produce a number of data files that are used
by the next command.

A mean b0 DWI image is calculated from all b0 volumes in the DTI dataset.

.. code-block:: python

    import tempfile
    import numpy as np
    from imagedata import Series
    from imagedata_registration.FSL import topup, apply_topup, bet, eddy

    def _calculate_ap_b0(self, ap: Series) -> Series:
        """Calculate mean B0 image"""
        ap_b0 = []
        tags = ap.tags[0]
        for idx in np.ndindex(tags.shape):
            try:
                b, bvector = tags[idx][0]
            except TypeError:
                continue
            if b == 0:
                ap_b0.append(ap[idx])
        ap_b0 = Series(np.mean(np.array(ap_b0), axis=0), template=ap, geometry=ap)
        return ap_b0

    pa = Series('DTI_6dir.zip?DTI_6dir_PA', dtype=float)
    ap = Series('DTI_6dir.zip?DTI_6dir_AP',
                'dti', dtype=float, accept_duplicate_tag=True)
    ap_b0 = self._calculate_ap_b0(ap)
    with tempfile.TemporaryDirectory() as workdir:
        ap_fieldcoef = topup(workdir, ap_b0, pa)
        ap_corrected = apply_topup(workdir, ap)
        bet_dwi = bet(workdir, ap_corrected[0], mask=True, skull=True)
        eddy_res = eddy(workdir, ap_corrected, mask=bet_dwi['mask'],
                        topup=os.path.join(workdir, 'AP_PA_topup'),
                        niter=1, repol=True, mporder=8
                        )
        eddy_corrected = eddy_res['out_corrected']
