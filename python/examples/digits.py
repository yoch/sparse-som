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
clf = SomClassifier(BSom) #, H, W, N)
#clf.setup(h=12, w=15)

# train SOM, do calibration and predict labels
y = clf.fit_predict(X, labels=dataset.target)

print(clf)
print('Quantization Error: %2.4f' % clf.quant_error)
print('Topographic  Error: %2.4f' % clf.topog_error)
print('='*50)
print(classification_report(dataset.target, y))
