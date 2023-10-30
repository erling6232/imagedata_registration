.. _SimpleElastix:

SimpleElastix Examples
======================

A function `register_elastix` is provided here.
This function will register a **moving** `Series` to a **fixed** `Series`.
`register_elastix` is based on one of the `SimpleElastix` examples
in
https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod1_docs.html
and can serve as an example for using `ITK/Elastix` methods.


.. code-block:: python

    from imagedata import Series
    from imagedata_registration.Elastix import register_elastix

    # fixed can be either a Series volume,
    # or an index (int) into moving Series
    # moving can be a 3D or 4D Series instance
    moving = Series("data/", "time")
    fixed = 10
    out = register_fsl(fixed, moving)


Documentation on ITK / Elastix
------------------------------
* SimpleElastix: https://simpleelastix.readthedocs.io/
* SimpleITK: https://simpleitk.readthedocs.io/
