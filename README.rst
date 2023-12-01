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
Each `FSL` method may have different requirements.
The common factor is that `FSL` methods will read and write NIfTI image files.

A function `register_fsl` is provided here.
This function will register a moving Series to a fixed Series.
The default registration method is fsl.MCFLIRT.
The function will accept other registration methods.

ITK Elastix
-----------

The popular `Elastix` GUI is based on the C++ `ITK` image registration routines.
Like the `FSL` methods, there are numerous `Elastix` methods available, all with
different requirements.
The `SimpleElastix`
(https://simpleelastix.readthedocs.io/index.html)
Python library is one particular interface to the `Elastix/ITK` routines.

Prerequisites
#############

NPreg on CUDA GPU
-----------------

imagedata-registration will benefit from a CUDA GPU. If this is available,
install `CuPy` (https://docs.cupy.dev).

* First, install the `CUDA Toolkit`: see https://developer.nvidia.com/cuda-toolkit.

* There are different options for installing `CuPy`. See:
  https://docs.cupy.dev/en/stable/install.html

FSL
---

The imagedata-registration FSL module is a wrapper around the official FSL tools.
A native FSL installation is required on the host computer.

SimpleElastix
-------------

SimpleElastix must be installed separately:

.. code-block::

    pip install SimpleITK-SimpleElastix

Installation
############

.. code-block::

    pip install imagedata-registration

Examples
########

`NPreg examples <docs/NPreg.rst>`_

`FSL examples <docs/FSL.rst>`_

`SimpleElastix examples <docs/SimpleElastix.rst>`_


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

