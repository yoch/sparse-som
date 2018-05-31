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
