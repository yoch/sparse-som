.. sparse_som documentation master file, created by
   sphinx-quickstart on Sun Nov 27 18:35:44 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :description: sparse-som: efficient Self-Organizing Map (SOM, Kohonen map) for sparse, high-dimensional data (e.g. text) in Python, with online and batch (OpenMP) algorithms.
   :keywords: self-organizing map, SOM, Kohonen, sparse data, libsvm, OpenMP, clustering, unsupervised learning, Python

************************
sparse-som documentation
************************

Python package for the efficient Self-Organizing Map (SOM, Kohonen map)
implementation of sparse-som, intended for sparse, high-dimensional data
(e.g. text).

* Install: ``pip install sparse-som`` (`PyPI <https://pypi.org/project/sparse-som/>`_)
* Source code and issue tracker: `GitHub <https://github.com/yoch/sparse-som>`_
* Paper: `Efficient Implementation of Self-Organizing Map for Sparse Input Data
  <https://doi.org/10.5220/0006499500540063>`_ (IJCCI 2017)


.. toctree::
   :maxdepth: 2

Module contents
===============


.. automodule:: sparse_som


enums
-----

.. autoclass:: sparse_som.cooling

    .. autoattribute:: sparse_som.cooling.LINEAR
    .. autoattribute:: sparse_som.cooling.EXPONENTIAL


.. autoclass:: sparse_som.topology

    .. autoattribute:: sparse_som.topology.CIRC
    .. autoattribute:: sparse_som.topology.HEXA
    .. autoattribute:: sparse_som.topology.RECT


classes
-------

*Self-Organizing Maps wrappers for python, intended for sparse input data.*

.. autoclass:: sparse_som.BSom
    :members: codebook, nrows, ncols, dim, umatrix

    .. automethod:: train(data, epochs=10, r0=0, rN=0.5, std=0.3, cool=cooling.LINEAR)
    .. automethod:: bmus(data, tg_error=False, qt_error=False)
    .. automethod:: activation_map(data)


.. autoclass:: sparse_som.Som
    :members: codebook, nrows, ncols, dim, umatrix

    .. automethod:: train(data, tmax, r0=0, a0=0.5, rN=0.5, aN=0., std=0.3, rcool=cooling.LINEAR, acool=cooling.LINEAR)
    .. automethod:: bmus(data, tg_error=False, qt_error=False)
    .. automethod:: activation_map(data)



Submodules
==========


sparse_som.classifier module
----------------------------


.. autoclass:: sparse_som.SomClassifier
    :members:

    .. automethod:: __init__



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
