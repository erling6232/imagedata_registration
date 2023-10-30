.. _FSL:

FSL Examples
============

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

