# sparse-som

Efficient Implementation of Self-Organizing Map for Sparse Input Data.

#### Main features

- Highly optimized for sparse data (LIBSVM format).
- Support both online and batch SOM algorithms.
- Parallel batch implementation (OpenMP).
- OS independent.
- Pyhton 3 support.

## Build

The simplest way to build the cli tools from the main directory : `cd src && make all`. After the compilation terminates, the resulting executables may be found in the `build` directory.

GCC is reccomended, but you can use another compiler if you want. C++11 support is required.

## Install

...

## Usage

### CLI

#### sparse-som

To use the *online* version :

```
Usage: ./sparse-som
        -i infile - input file at libsvm sparse format
        -y nlig - number of lines in the codebook
        -x ncol - number of columns in the codebook
        [ -u ] - one based column indices (default is zero based)
        [ -N ] - normalize the input vectors
        [ -l codebook ]   - load codebook from binary file
        [ -o|O codebook ] - output codebook to filename (o:binary, O:text)
        [ -c|C classes ]  - output classification (c:without counts, C:with counts)
        [ -t tmax | -T epochs ] - number of training iterations (epoch=nb. of samples)
        [ -n neighborhood ] - neighborhood topology: 4=circ, 6=hexa, 8=rect (default 8)
        [ -r radius0 -R radiusN ] - radius at start and end (default r=(x+y)/2, R=0.5)
        [ -a alpha0  -A  alphaN ] - learning rate at start and end (default a=0.5, A=1.e-37)
        [ -H radiusCool ] - radius cooling: 0=linear, 1=exponential (default 0)
        [ -h  alphaCool ] -  alpha cooling: 0=linear, 1=exponential (default 0)
        [ -s stdCoeff ] - sigma = radius * coeff (default 0.3)
        [ -q ] - quiet

```

#### sparse-bsom

To use the *batch* version :

```
Usage: ./sparse-bsom
        -i infile - input file at libsvm sparse format
        -y nlig - number of lines in the codebook
        -x ncol - number of columns in the codebook
        [ -u ] - one based column indices (default is zero based)
        [ -N ] - normalize the input vectors
        [ -l codebook ]   - load codebook from binary file
        [ -o|O codebook ] - output codebook to filename (o:binary, O:text)
        [ -c|C classes ]  - output classification (c:without counts, C:with counts)
        [ -T epochs ] - number of epochs (default 10)
        [ -n neighborhood ] - neighborhood topology: 4=circ, 6=hexa, 8=rect (default 8)
        [ -r radius0 -R radiusN ] - radius at start and end (default r=(x+y)/2, R=0.5)
        [ -H radiusCool ] - radius cooling: 0=linear, 1=exponential (default 0)
        [ -s stdCoeff ] - sigma = radius * coeff (default 0.3)
        [ -q ] - quiet
```

### Python


```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sparse_som import *

# load some dataset
dataset = load_digits()

# convert to sparse CSR format
X = csr_matrix(dataset.data)

# setup SOM dimensions
H, W = 12, 15   # Network height and width
N = X.shape[1]  # Nb. features (vectors dimension)

################ Simple usage ################

# setup SOM network
som = Som(H, W, N, topology.HEXA) # , verbose=True
print(som.codebook.shape)

# reinit the codebook (not needed)
som.codebook = np.random.rand(H, W, N).\
                astype(som.codebook.dtype, copy=False)

# train the SOM
som.train(X)

################ Use classifier ################

# setup SOM classifier (using batch SOM)
cls = SomClassifier(BSom, H, W, N)

# use SOM calibration
cls.fit(X, labels=dataset.target)

# make predictions
y = cls.predict(X)

print(classification_report(dataset.target, y))
```

## Documentation

...
