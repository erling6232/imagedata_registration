.. _FSL:

FSL Examples
============

Using MCFLIRT module:

.. code-block:: python

    from imagedata_registration.FSL import register_fsl
    import nipype.interfaces.fsl as fsl

    # fixed can be either a Series volume,
    # or an index (int) into moving Series
    # moving can be a 3D or 4D Series instance
    out = register_fsl(fixed, moving, method=fsl.MCFLIRT)
    out.seriesDescription += " (MCFLIRT)"

