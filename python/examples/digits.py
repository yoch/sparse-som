import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sparse_som import *

# Load some dataset
dataset = load_digits()

# convert to sparse CSR format
X = csr_matrix(dataset.data)

# setup SOM dimensions
H, W = 12, 15   # Network height and width
N = X.shape[1]  # Nb. features (vectors dimension)

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

# use SOM calibration
cls.fit(X, labels=dataset.target)

# make predictions
y = cls.predict(X)

print(classification_report(dataset.target, y))
