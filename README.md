# sparse-som

Efficient Implementation of Self-Organizing Map for Sparse Input Data.

This program uses an algorithm especially intended for sparse data, 
which much faster than the classical one on very sparse datasets
(time-complexity depend to non-zero values only).

#### Main features

- Highly optimized for sparse data (LIBSVM format).
- Support both online and batch SOM algorithms.
- Parallel batch implementation (OpenMP).
- OS independent.
- [Python](https://pypi.python.org/pypi?:action=display&name=sparse-som) support.

## Build

The simplest way to build the cli tools from the main directory : `cd src && make all`.
After the compilation terminates, the resulting executables may be found in the `build` directory.

GCC is reccomended, but you can use another compiler if you want. C++11 support is required.
OpenMP support is required to take advantage of parallelism (sparse-bsom).


## Install

####

No install required.

#### Python

To install the python version, simply run `pip install sparse-som`.


## Usage

### CLI

#### sparse-som

To use the *online* version :

```
Usage: sparse-som
        -i infile        input file at libsvm sparse format
        -y nrows         number of rows in the codebook
        -x ncols         number of columns in the codebook
        [ -d dim ]       force the dimension of codebook's vectors
        [ -u ]           one based column indices (default is zero based)
        [ -N ]           normalize the input vectors
        [ -l cb ]        load codebook from binary file
        [ -o|O cb ]      output codebook to filename (o:binary, O:text)
        [ -c|C cl ]      output classification (c:without counts, C:with counts)
        [ -n neig ]      neighborhood topology: 4=circ, 6=hexa, 8=rect (default 8)
        [ -t n | -T e ]  number of training iterations or epochs (epoch = nrows)
        [ -r r0 -R rN ]  radius at start and end (default r=(x+y)/2, R=0.5)
        [ -a a0 -A aN ]  learning rate at start and end (default a=0.5, A=1.e-37)
        [ -H rCool ]     radius cooling: 0=linear, 1=exponential (default 0)
        [ -h aCool ]     alpha cooling: 0=linear, 1=exponential (default 0)
        [ -s stdCf ]     sigma = radius * stdCf (default 0.3)
        [ -v ]           increase verbosity level (default 0, max 2)
```

#### sparse-bsom

To use the *batch* version :

```
Usage: sparse-bsom
        -i infile        input file at libsvm sparse format
        -y nrows         number of rows in the codebook
        -x ncols         number of columns in the codebook
        [ -d dim ]       force the dimension of codebook's vectors
        [ -u ]           one based column indices (default is zero based)
        [ -N ]           normalize the input vectors
        [ -l cb ]        load codebook from binary file
        [ -o|O cb ]      output codebook to filename (o:binary, O:text)
        [ -c|C cl ]      output classification (c:without counts, C:with counts)
        [ -n neig ]      neighborhood topology: 4=circ, 6=hexa, 8=rect (default 8)
        [ -T epoc ]      number of epochs (default 10)
        [ -r r0 -R rN ]  radius at start and end (default r=(x+y)/2, R=0.5)
        [ -H rCool ]     radius cooling: 0=linear, 1=exponential (default 0)
        [ -s stdCf ]     sigma = radius * stdCf (default 0.3)
        [ -v ]           increase verbosity level (default 0, max 2)
```

To control the number of threads used by OpenMP, set to `OMP_NUM_THREADS` variable to the desired value, for example :

```
OMP_NUM_THREADS=4 sparse-bsom ...
```

If undefined one thread per CPU is used.

### Python


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

Other examples are available in the `python/examples` directory.


## Documentation

### CLI

#### Files Format

Input files must be at LIBSVM format.

```
<label> <index1>:<value1> <index2>:<value2> ...
.
.
.
```

Each line contains an instance and is ended by a '\n' character. The pair `<index>:<value>` gives a feature (attribute) value: `<index>` is an integer starting from 0 and `<value>` is a real number. Indices must be in ASCENDING order. Labels in the file are only used for network calibration. If they are unknown, just fill the first column with any numbers.


### Python documentation

The python documentation can be found at: http://sparse-som.readthedocs.io/en/latest/


### API

The C++ API is not public yet, because things still may change.


## How to cite this work

```
@InProceedings{melka-mariage:ijcci17,
  author={Melka, Josu{\'e} and Mariage, Jean-Jacques},
  title={Efficient Implementation of Self-Organizing Map for Sparse Input Data},
  booktitle={Proceedings of the 9th International Joint Conference on Computational Intelligence: IJCCI},
  volume={1},
  month={November},
  year={2017},
  address={Funchal, Madeira, Portugal},
  pages={54-63},
  publisher={SciTePress},
  organization={INSTICC},
  doi={10.5220/0006499500540063},
  isbn={978-989-758-274-5},
  url={http://www.ai.univ-paris8.fr/~jmelka/IJCCI_2017_20.pdf}
}
```

```
@Inbook{Melka2019,
    author    = "Melka, Josu{\'e} and Mariage, Jean-Jacques",
    editor    = "Sabourin, Christophe and Merelo, Juan Julian and Madani, Kurosh and Warwick, Kevin",
    title     = "Adapting Self-Organizing Map Algorithm to Sparse Data",
    bookTitle = "Computational Intelligence: 9th International Joint Conference, IJCCI 2017 Funchal-Madeira, Portugal, November 1-3, 2017 Revised Selected Papers",
    year      = "2019",
    publisher = "Springer International Publishing",
    address   = "Cham",
    pages     = "139--161",
    isbn      = "978-3-030-16469-0",
    doi       = "10.1007/978-3-030-16469-0_8",
    url       = "https://doi.org/10.1007/978-3-030-16469-0_8"
}
```

