import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_mldata
from sparse_som import *

# Load some dataset
dataset = fetch_mldata('MNIST original')

# convert to sparse CSR format
X = csr_matrix(dataset.data, dtype=np.float32)
X /= 255    # scale to 0 - 1 range

# setup SOM dimensions
H, W = 12, 16   # Network height and width
_, N = X.shape  # Nb. features (vectors dimension)

# setup SOM network
som = Som(H, W, N) # , verbose=True

# train the SOM
som.train(X, tmax=10**6)

# Plot the map
SHAPE = (28, 28)
mapping = np.vstack([np.hstack([node.reshape(SHAPE)
                                for node in row])
                     for row in som.codebook])
plt.matshow(mapping, cmap='gray')
plt.show()
