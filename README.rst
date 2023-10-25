#################################
Image registration with imagedata
#################################

|Docs Badge| |buildstatus|  |coverage| |pypi|


Helper modules to do
image registration for `Imagedata` **Series** objects.

Available modules
#################

NPreg
-----

`NPreg` by Erlend Hodneland is implemented in Python,
and available as a self-supported PyPi package.
There are three implementations of `NPreg`:

* Pure Python/NumPy code. Source code will run on any Python platform.
* Cython code. Binary code compiled for supported platforms.
* CuPy/CUDA code. Source code which will run on platforms with a working `CuPy` and CUDA Toolkit.

FSL
---

`FSL`
(https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL)
has several methods for image registration.
Using `FSL` image registration from Python requires the `FSL` interface from
nipype, as well as a the `FSL` executables.
Each `FSL` may have different requirements.
The common factor is that `FSL` will read and write NIfTI image files.

Rather than providing an interface between `Imagedata`and `FSL`,
a skeleton for a program is suggested.

ITK Elastix
-----------

The popular `Elastix` GUI is based on the C++ `ITK` image registration routines.
Like the `FSL` methods, there are numerous `ITK` methods available, all with
different requirements.
The `SimpleElastix`
(https://simpleelastix.readthedocs.io/index.html)
Python library is one particular interface to the `Elastix/ITK` routines.

Rather than providing an interface between `Imagedata`and `SimpleElastix`,
a skeleton for a program is suggested.

Prerequisites
#############

NPreg on CUDA GPU
-----------------

imagedata-registration will benefit from a CUDA GPU. If this is available,
install `CuPy` (https://docs.cupy.dev).

First, install the `CUDA Toolkit`: see https://developer.nvidia.com/cuda-toolkit.

There are different options for installing `CuPy`. See:
https://docs.cupy.dev/en/stable/install.html

FSL
---

The imagedata-registration FSL module is a wrapper around the official FSL tools.
A native FSL installation is required on the host computer.

Installation
############

.. code-block::

    pip install imagedata-registration

Examples
########

NPreg
-----

See [NPreg examples](docs/NPreg.rst).
See :doc:`docs/NPreg.rst`
Using NPreg module:

.. code-block:: python

    from imagedata_registration.NPreg import register_npreg
    from imagedata_registration.NPreg.multilevel import CYCLE_NONE, CYCLE_V2

    # fixed can be either a Series volume,
    # or an index (int) into moving Series
    # moving can be a 3D or 4D Series instance
    out = register_npreg(fixed, moving, cycle=CYCLE_NONE)
    out.seriesDescription += " (NPreg)"

FSL
---

Using MCFLIRT module:

.. code-block:: python

    from imagedata_registration.FSL import register_fsl
    import nipype.interfaces.fsl as fsl

    # fixed can be either a Series volume,
    # or an index (int) into moving Series
    # moving can be a 3D or 4D Series instance
    out = register_fsl(fixed, moving, method=fsl.MCFLIRT)
    out.seriesDescription += " (MCFLIRT)"


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

