#################################
Image registration with imagedata
#################################

|Docs Badge| |buildstatus|  |coverage| |pypi|


Image registration routines for Imagedata.

Available modules:

   * NPreg

Installation
------------

.. code-block::

    pip install imagedata-registration

Example
-------

Using NPreg module:

.. code-block:: python

    from imagedata_registration.NPreg import register_series

    # fixed can be either a Series volume,
    # or an index (int) into moving Series
    # moving can be a 3D or 4D Series instance
    out = register_series(fixed, moving)
    out.seriesDescription += " (NPreg)"

.. |Docs Badge| image:: https://readthedocs.org/projects/imagedata_registration/badge/
    :alt: Documentation Status
    :scale: 100%
    :target: https://imagedata_registration.readthedocs.io

.. |buildstatus| image:: https://github.com/erling6232/imagedata_registration/actions/workflows/build_wheels.yml/badge.svg
    :target: https://github.com/erling6232/imagedata_registration/actions?query=branch%3Amain
    :alt: Build Status

.. _buildstatus: https://github.com/erling6232/imagedata_registration/actions

.. |coverage| image:: https://codecov.io/gh/erling6232/imagedata_registration/branch/main/graph/badge.svg?token=1OPGNXJ8Z3
    :alt: Coverage
    :target: https://codecov.io/gh/erling6232/imagedata_registration

.. |pypi| image:: https://img.shields.io/pypi/v/imagedata-registration.svg
    :target: https://pypi.python.org/pypi/imagedata-registration
    :alt: PyPI Version

