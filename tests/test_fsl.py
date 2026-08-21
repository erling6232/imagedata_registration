#!/usr/bin/env python3

import os
import unittest
import numpy as np
import tempfile
from imagedata import Series  # , Study

from imagedata_registration.FSL import register_fsl, topup, apply_topup, bet, eddy


class TestFSLRegistration(unittest.TestCase):
    def test_register_fsl(self):
        if os.getenv("GITHUB_ACTION") is not None:
            return
        a = Series('data/time.zip', 'time')
        out = register_fsl(0, a, options={"cost": "corratio"})
        np.testing.assert_array_equal(out.tags[0], a.tags[0])
        self.assertEqual(out.axes[0], a.axes[0])
        with tempfile.TemporaryDirectory() as d:
            out.write(d, formats=['dicom'])


class TestFSLDWI(unittest.TestCase):

    def setUp(self) -> None:
        os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

    def _calculate_ap_b0(self, ap: Series) -> Series:
        ap_b0 = []
        tags = ap.tags[0]
        for idx in np.ndindex(tags.shape):
            try:
                if ap.input_order == 'dti':
                    b, bvector = tags[idx][0]
                else:
                    b, bvector = tags[idx]
            except TypeError:
                continue
            if b == 0:
                ap_b0.append(ap[idx])
        ap_b0 = Series(np.mean(np.array(ap_b0), axis=0), template=ap, geometry=ap)
        return ap_b0

    def test_topup(self):
        pa = Series(os.path.join('data', 'DTI_6dir.zip?DTI_6dir_PA'), dtype=float)
        ap = Series(os.path.join('data', 'DTI_6dir.zip?DTI_6dir_AP'),
                    'b,bvector', dtype=float, accept_duplicate_tag=True)
                    # 'dti', dtype=float, accept_duplicate_tag=True)
        ap_b0 = self._calculate_ap_b0(ap)
        with tempfile.TemporaryDirectory() as tmp:
            print(f'Working directory: {tmp}')
            pa.write(os.path.join(tmp, 'pa.nii.gz'), formats=['nifti'])
            ap.write(os.path.join(tmp, 'ap.nii.gz'), formats=['nifti'])
            fieldcoef = topup(tmp, ap_b0, pa)

            self.assertEqual(fieldcoef.shape, (9, 67, 67))

            ap_corrected = apply_topup(tmp, ap)
            self.assertEqual(ap_corrected.shape, ap.shape)

    def test_bet(self):
        dwi = Series(os.path.join('data', 'DTI_6dir.zip?DTI_6dir_PA'), dtype=float)
        with tempfile.TemporaryDirectory() as tmp:
            bet_dwi = bet(tmp, dwi, mask=True, skull=True)
        mask = bet_dwi['mask']
        self.assertEqual(mask.shape, dwi.shape)
        np.testing.assert_array_equal(mask.spacing, dwi.spacing)

    def test_eddy(self):
        pa = Series(os.path.join('data', 'DTI_6dir.zip?DTI_6dir_PA'), dtype=float)
        ap = Series(os.path.join('data', 'DTI_6dir.zip?DTI_6dir_AP'),
                    'dti', dtype=float, accept_duplicate_tag=True)
        ap_b0 = self._calculate_ap_b0(ap)
        with tempfile.TemporaryDirectory() as tmp:
            ap_fieldcoef = topup(tmp, ap_b0, pa)
            ap_corrected = apply_topup(tmp, ap)
            bet_dwi = bet(tmp, ap_corrected, mask=True, skull=True)
            eddy_dwi = eddy(tmp, ap_corrected, mask=bet_dwi['mask'],
                            topup=os.path.join(tmp, 'AP_PA_topup')
                            )
        pass


if __name__ == '__main__':
    unittest.main()
