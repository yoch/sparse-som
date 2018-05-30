import pytest
import numpy as np
import scipy.sparse as sp
from sparse_som import *



@pytest.fixture
def random_matrix():
    return sp.random(200, 1000, density=0.01, format='csr', dtype='f')


#TODO: these tests are non-fail tests only, add functional tests


########################## SOM ##########################

def test_som_creation():
    som = Som(10, 15, 1000)
    assert som.nrows == 10
    assert som.ncols == 15
    assert som.dim == 1000

def test_som_codebook_getter():
    som = Som(10, 15, 1000)
    assert som.codebook.shape == (10, 15, 1000)

def test_som_codebook_setter():
    som = Som(10, 15, 1000)
    cb = np.random.rand(10, 15, 1000)
    som.codebook = cb
    assert np.array_equal(som.codebook, cb)

@pytest.mark.parametrize("std", [0.3, 0.5, 1.])
@pytest.mark.parametrize("r0,rN", [(15, 0.5), (10, 0)])
@pytest.mark.parametrize("a0,aN", [(1, 0), (0.5, 0.0001)])
def test_som_train(random_matrix, std, r0, rN, a0, aN):
    n, d = random_matrix.shape
    som = Som(10, 15, d)
    som.train(random_matrix, tmax=n * 10, std=std, r0=r0, rN=rN, a0=a0, aN=aN)

@pytest.mark.parametrize("topol", [topology.RECT, topology.HEXA])
def test_som_train_topol(random_matrix, topol):
    n, d = random_matrix.shape
    som = Som(10, 15, d, topol=topol)
    som.train(random_matrix, tmax=n * 10)

@pytest.mark.parametrize("epochs", [0, 1, 10])
def test_som_train_epochs(random_matrix, epochs):
    n, d = random_matrix.shape
    tmax = n * epochs
    som = Som(10, 15, d)
    som.train(random_matrix, tmax=tmax)

@pytest.mark.parametrize("rcool", [cooling.LINEAR, cooling.EXPONENTIAL])
@pytest.mark.parametrize("acool", [cooling.LINEAR, cooling.EXPONENTIAL])
def test_som_train_cooling(random_matrix, rcool, acool):
    n, d = random_matrix.shape
    som = Som(10, 15, d)
    som.train(random_matrix, tmax=n * 10, rcool=rcool, acool=acool)

def test_som_bmus(random_matrix):
    n, d = random_matrix.shape
    som = Som(10, 15, d)
    som.train(random_matrix, tmax=n * 5)
    assert som.bmus(random_matrix).shape == (n, 2)


########################## BSOM ##########################

def test_bsom_creation():
    som = BSom(10, 15, 1000)
    assert som.nrows == 10
    assert som.ncols == 15
    assert som.dim == 1000

def test_bsom_codebook_getter():
    som = BSom(10, 15, 1000)
    assert som.codebook.shape == (10, 15, 1000)

def test_bsom_codebook_setter():
    som = BSom(10, 15, 1000)
    cb = np.random.rand(10, 15, 1000).astype('f')
    som.codebook = cb
    assert np.array_equal(som.codebook, cb)

@pytest.mark.parametrize("std", [0.3, 0.5, 1.])
@pytest.mark.parametrize("r0,rN", [(15, 0.5), (10, 0)])
def test_bsom_train(random_matrix, std, r0, rN):
    n, d = random_matrix.shape
    som = BSom(10, 15, d)
    som.train(random_matrix, epochs=10, std=std, r0=r0, rN=rN)

@pytest.mark.parametrize("epochs", [0, 1, 10])
def test_bsom_train_epochs(random_matrix, epochs):
    n, d = random_matrix.shape
    som = BSom(10, 15, d)
    som.train(random_matrix, epochs=epochs)

@pytest.mark.parametrize("topol", [topology.RECT, topology.HEXA])
def test_bsom_train_topol(random_matrix, topol):
    n, d = random_matrix.shape
    som = BSom(10, 15, d, topol=topol)
    som.train(random_matrix, epochs=10)

@pytest.mark.parametrize("cool", [cooling.LINEAR, cooling.EXPONENTIAL])
def test_bsom_train_cooling(random_matrix, cool):
    n, d = random_matrix.shape
    som = BSom(10, 15, d)
    som.train(random_matrix, epochs=10, cool=cool)

def test_bsom_bmus(random_matrix):
    n, d = random_matrix.shape
    som = BSom(10, 15, d)
    som.train(random_matrix, epochs=5)
    assert som.bmus(random_matrix).shape == (n, 2)

#TODO: test classifier as well

from sklearn.metrics import f1_score

@pytest.fixture(scope="module")
def digits():
    from sklearn.datasets import load_digits
    digits = load_digits()
    digits.data = sp.csr_matrix(digits.data)
    return digits


@pytest.mark.parametrize("cls", [Som, BSom])
def test_som_classifier_fit(digits, cls):
    _, d = digits.data.shape
    clf = SomClassifier(cls, 5, 5, d)
    clf.fit(digits.data, digits.target)

@pytest.mark.parametrize("cls", [Som, BSom])
def test_som_classifier_fit_predict(digits, cls):
    _, d = digits.data.shape
    clf = SomClassifier(cls, 7, 9, d)
    clf.fit(digits.data[:1000], digits.target[:1000])
    y = clf.predict(digits.data[1000:], unkown=-1)
    f = (y != -1) #mask: skip unkowns
    assert f1_score(y[f], digits.target[1000:][f], average='weighted') >= 0.9
