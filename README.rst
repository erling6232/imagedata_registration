#################################
Image registration with imagedata
#################################

|Docs Badge| |buildstatus|  |coverage| |pypi|


Image registration routines for Imagedata.

Available modules
#################

NPreg
-----

There are three implementations of NPreg:

* Pure Python/NumPy code. Source code will run on any Python platform.
* Cython code. Binary code compiled for supported platforms.
* CuPy/CUDA code. Source code which will run on platforms with a working `CuPy` and CUDA Toolkit.

Prerequisites
#############

imagedata-registration will benefit from a CUDA GPU. If this is available,
install `CuPy` (https://docs.cupy.dev):

Install CUDA Toolkit: see https://developer.nvidia.com/cuda-toolkit.

There are different options for installing `CuPy`:
https://docs.cupy.dev/en/stable/install.html

Installation
############

.. code-block::

    pip install imagedata-registration

Example
#######

Using NPreg module:

.. code-block:: python

    from imagedata_registration.NPreg import register_npreg
    from imagedata_registration.NPreg.multilevel import CYCLE_NONE, CYCLE_V2

    # fixed can be either a Series volume,
    # or an index (int) into moving Series
    # moving can be a 3D or 4D Series instance
    out = register_npreg(fixed, moving, cycle=CYCLE_NONE)
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

