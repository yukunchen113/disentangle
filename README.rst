========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/disentangle/badge/?style=flat
    :target: https://readthedocs.org/projects/disentangle
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/yukunchen113/disentangle.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/yukunchen113/disentangle

.. |requires| image:: https://requires.io/github/yukunchen113/disentangle/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/yukunchen113/disentangle/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/yukunchen113/disentangle/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/yukunchen113/disentangle

.. |version| image:: https://img.shields.io/pypi/v/disentangle.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/disentangle

.. |wheel| image:: https://img.shields.io/pypi/wheel/disentangle.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/disentangle

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/disentangle.svg
    :alt: Supported versions
    :target: https://pypi.org/project/disentangle

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/disentangle.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/disentangle

.. |commits-since| image:: https://img.shields.io/github/commits-since/yukunchen113/disentangle/v0.0.4.svg
    :alt: Commits since latest release
    :target: https://github.com/yukunchen113/disentangle/compare/v0.0.4...master



.. end-badges

Quickly create disentanglement pipelines for machine learning research.

This has prebuilt and functionality for tensorflow function wrappers, to make changing the architecture easy. Though it is not too hard to use keras, viewing the whole architecture on one page and making comparisons between different archtectures might be a bit of a pain and prone to errors. 

Though this library can be used to wrap any tensorflow (or custom) keras module, making it useful for a variety of different fields of deep learning research. This library is called disentangle as my main focus of research is on disentanglement and much of the functionality (such as visualization tools, prebuilt architectures, etc) are biased towards that.

There are a few parts of this which use functionality from disentanglement_lib, so please take a look at their work. This code is contained in other_library_tools.

Other than tensorflow wrappers, this library contains useful functionality for datasets (custom, if needed), visualizations, and general useful tools such as timers and path managers.

Thanks to ionelmc's cookiecutter-pylibrary for the generation of a pypi library with many good practices. So I could easily convert this library to be deployable to pypi.

* Free software: MIT license

Installation
============

::

    pip install disentangle

You can also install the in-development version with::

    pip install https://github.com/yukunchen113/disentangle/archive/master.zip


Documentation
=============


https://disentangle.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
