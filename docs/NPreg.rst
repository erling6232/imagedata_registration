.. _NPreg:

NPreg Examples
==============

Using NPreg module:


.. code-block:: python

    from imagedata import Series
    from imagedata_registration.NPreg import register_npreg
    from imagedata_registration.NPreg.multilevel import CYCLE_NONE, CYCLE_V2

    # fixed can be either a Series volume,
    # or an index (int) into the moving Series
    # moving can be a 3D or 4D Series instance
    moving = Series("data/", "time")
    fixed = 10
    out = register_npreg(fixed, moving, cycle=CYCLE_NONE)

