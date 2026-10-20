# sparse-som

[![PyPI](https://img.shields.io/pypi/v/sparse-som.svg)](https://pypi.org/project/sparse-som/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/yoch/sparse-som/blob/master/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5220%2F0006499500540063-blue.svg)](https://doi.org/10.5220/0006499500540063)

Python package for the efficient Self-Organizing Map (SOM, Kohonen map) implementation of
[sparse-som](https://github.com/yoch/sparse-som), intended for sparse, high-dimensional data (e.g. text).

It uses an algorithm especially intended for sparse data, which is much faster than the
classical one on very sparse datasets (time complexity depends on non-zero values only).

## Main features

- Highly optimized for sparse data (`scipy.sparse.csr_matrix` input).
- Both online (`Som`) and batch (`BSom`) SOM algorithms.
- Parallel batch implementation (OpenMP).
- `SomClassifier` to calibrate a trained map with labels and predict.

## Install

```
pip install sparse-som
```

Building from source requires a C++11 compiler with OpenMP support; `pip` installs the build
requirements (Cython 3, numpy) itself, see `pyproject.toml`.

## Usage

```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sparse_som import *

# Load some dataset
dataset = load_digits()

# convert to sparse CSR format
X = csr_matrix(dataset.data, dtype=np.float32)

# setup SOM dimensions
H, W = 12, 15   # Network height and width
_, N = X.shape  # Nb. features (vectors dimension)

################ Simple usage ################

# setup SOM network
som = Som(H, W, N, topology.HEXA) # , verbose=True
print(som.nrows, som.ncols, som.dim)

# reinit the codebook (not needed)
som.codebook = np.random.rand(H, W, N).\
                    astype(som.codebook.dtype, copy=False)

# train the SOM
som.train(X)

# get bmus for the data
bmus = som.bmus(X)

################ Use classifier ################

# setup SOM classifier (using batch SOM)
cls = SomClassifier(BSom, H, W, N)

# train SOM, do calibration and predict labels
y = cls.fit_predict(X, labels=dataset.target)

print('Quantization Error: %2.4f' % cls.quant_error)
print('Topographic  Error: %2.4f' % cls.topog_error)
print('='*50)
print(classification_report(dataset.target, y))
```

Other examples are available in the
[`python/examples`](https://github.com/yoch/sparse-som/tree/master/python/examples) directory.

## Links

- Documentation: https://sparse-som.readthedocs.io/
- Source code, command-line tools and issue tracker: https://github.com/yoch/sparse-som
- Paper: https://doi.org/10.5220/0006499500540063

## How to cite

Melka, J. and Mariage, J.-J. (2017). Efficient Implementation of Self-Organizing Map for Sparse Input Data.
In *Proceedings of the 9th International Joint Conference on Computational Intelligence: IJCCI*, vol. 1, pp. 54-63.
SciTePress. https://doi.org/10.5220/0006499500540063

BibTeX entries (including the extended 2019 Springer chapter, https://doi.org/10.1007/978-3-030-16469-0_8)
are in the [project README](https://github.com/yoch/sparse-som#how-to-cite-this-work).

## Development

### Cython

#### Compilation
`sparse_som/som.cpp` is generated from `som.pyx` at build time, and not tracked.
```
pip install "cython>=3.0.11" numpy setuptools
python setup.py build_ext --inplace
```

#### Cleanup
```
python setup.py clean
```

#### Build package
```
python -m build
```


### Build doc
autodoc needs the compiled package installed (Read the Docs does the same, see `.readthedocs.yaml`).
```
pip install . sphinx
sphinx-build -b html doc/source doc/build
```


### Running tests
```
python -m pytest tests.py
```
