.. _NPreg:

NPreg Examples
==============

Using NPreg module:


.. code-block:: python

    from imagedata_registration.NPreg import register_npreg
    from imagedata_registration.NPreg.multilevel import CYCLE_NONE, CYCLE_V2

    # fixed can be either a Series volume,
    # or an index (int) into the moving Series
    # moving can be a 3D or 4D Series instance
    out = register_npreg(fixed, moving, cycle=CYCLE_NONE)
    out.seriesDescription += " (NPreg)"

