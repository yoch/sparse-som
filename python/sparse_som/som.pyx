# distutils: language = c++
# distutils: sources = lib/bsom.cpp, lib/som.cpp, lib/sparse_vec.cpp

from libc.stdint cimport uint32_t, int32_t
from libc.stdlib cimport malloc, free
from libcpp cimport bool as bool_t
from libcpp.vector cimport vector
#from libcpp.string cimport string

cimport cython
cimport numpy as np
from cython cimport view
import numpy as np
import scipy.sparse



cdef extern from "lib/sparse_vec.h":
    cdef cppclass cell:
        uint32_t idx
        float val
    cdef cppclass sparse_vec:
        sparse_vec()
        void normalize()
        void clear()
        size_t size()
        void push_back(cell&)


## more generic, but too slow
#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function
#cdef sparse_matrix_from_scipy_old(vector[sparse_vec]& m, sm):
#    "convert scipy.sparse matrix to internal sparse representation"
#    cdef sparse_vec v
#    cdef cell c
#    cdef int32_t i, j, k, lasti = 0
#    cdef np.ndarray[int32_t, ndim=1] X
#    cdef np.ndarray[int32_t, ndim=1] Y
#    Y, X = sm.nonzero()
#    for k in range(len(Y)):
#        i = Y[k]
#        j = X[k]
#        c.idx = j
#        c.val = sm[i,j]
#        if i != lasti:
#            m.push_back(v)
#            lasti = i
#            v.clear()
#        v.push_back(c)
#        #print((i, j), c.val, v.size(), m.size())
#    m.push_back(v)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef sparse_matrix_from_scipy(vector[sparse_vec]& m, sm):
    "convert scipy.sparse.spmatrix to internal sparse representation"
    # sanity checks
    assert isinstance(sm, scipy.sparse.spmatrix)
    if not isinstance(sm, scipy.sparse.csr_matrix):
        sm = sm.tocsr()
    if not sm.has_sorted_indices:
        print("sorting indices")
        sm.sort_indices()
    # convert to internal representation
    cdef sparse_vec v
    cdef cell c
    cdef np.ndarray[float, ndim=1] data = sm.data.astype(np.float32, copy=False)
    cdef np.ndarray[int32_t, ndim=1] indices = sm.indices
    cdef np.ndarray[int32_t, ndim=1] indptr = sm.indptr
    cdef int32_t i, j
    for i in range(len(indptr)-1):
        for j in range(indptr[i], indptr[i+1]):
            c.idx = indices[j]
            c.val = data[j]
            v.push_back(c)
        m.push_back(v)
        v.clear()



cdef extern from "lib/som.h" namespace "som":
    cpdef enum cooling:
        LINEAR = 0
        EXPONENTIAL = 1

    cpdef enum topology:
        RECT = 8
        HEXA = 6
        CIRC = 4

    cdef cppclass _BSom "som::BSom":
        _BSom(size_t, size_t, size_t, topology, bool_t)
        void train(const vector[sparse_vec]&, size_t, float, float, float, cooling)
        void getBmus(const vector[sparse_vec]&, size_t *, float *)
        size_t getx()
        size_t gety()
        size_t getz()
        float * _codebookptr()

    cdef cppclass _Som "som::Som":
        _Som(size_t, size_t, size_t, topology, bool_t)
        void train(const vector[sparse_vec]&, size_t, double, double, double, double, double, cooling, cooling)
        size_t getx()
        size_t gety()
        size_t getz()
        double * _codebookptr()


cdef class BSom:
    """\
    Uses the batch algorithm and can take advantage
    from multi-core processors to learn efficiently.

    :param h: the network height
    :type h: int
    :param w: the network width
    :type w: int
    :param d: the dimension of input vectors
    :type d: int
    :param topol: the network topology
    :type topol: :const:`topology.RECT` or :const:`topology.HEXA`
    :param verbose: verbosity parameter
    :type verbose: bool
    """

    cdef _BSom * c_som

    def __cinit__(self, size_t h, size_t w, size_t d, topology topol=topology.RECT, bool_t verbose=False):
        self.c_som = new _BSom(h, w, d, topol, verbose)

    def __dealloc__(self):
        del self.c_som

    def train(self, data, size_t epochs=10, float r0=0, float rN=0.5, float std=0.3, cooling cool=cooling.LINEAR):
        """\
        Train the network with data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.float32`)
        :type data: :class:`scipy.sparse.spmatrix`
        :param epochs: number of epochs
        :type epochs: int
        :param r0: radius at the first iteration
        :type r0: float
        :param rN: radius at the last iteration
        :type rN: float
        :param cool: cooling strategy
        :type cool: :const:`cooling.LINEAR` or :const:`cooling.EXPONENTIAL`
        """
        if r0 == 0:
            r0 = min(self.c_som.gety(), self.c_som.getx()) / 2
        cdef vector[sparse_vec] m
        sparse_matrix_from_scipy(m, data)
        self.c_som.train(m, epochs, r0, rN, std, cool)

    def bmus(self, data):
        """\
        Return the best match units for data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.float32`)
        :type data: :class:`scipy.sparse.spmatrix`
        :returns: an array of the bmus coordinates (y,x)
        :rtype: 2D :class:`numpy.ndarray`
        """
        shape = self.c_som.gety(), self.c_som.getx()
        cdef vector[sparse_vec] m
        sparse_matrix_from_scipy(m, data)
        cdef size_t * bmus = <size_t*> malloc(m.size() * sizeof(size_t))
        cdef float * dsts = <float*> malloc(m.size() * sizeof(float))
        self.c_som.getBmus(m, bmus, dsts)
        cdef view.array arr = <size_t[:m.size()]> bmus
        YX = np.unravel_index(arr, shape)
        free(bmus)
        free(dsts)
        return np.vstack(YX).transpose()

    @property
    def codebook(self):
        """\
        :returns: a view of the internal codebook.
        :rtype: 3D :class:`numpy.ndarray`
        """
        y, x, z = self.c_som.gety(), self.c_som.getx(), self.c_som.getz()
        cdef float* data = self.c_som._codebookptr()
        cdef view.array arr = <float[:y,:x,:z]> data
        return np.asarray(arr)

    @codebook.setter
    def codebook(self, arr):
        """\
        Set the internal codebook.

        :param arr: the new codebook
        :type arr: 3D :class:`numpy.ndarray`
        """
        y, x, z = self.c_som.gety(), self.c_som.getx(), self.c_som.getz()
        assert (y, x, z) == arr.shape
        if arr.dtype != np.float32:
            print('changing codebook type to float32')
            arr = arr.astype(np.float32)
        cdef float* data = self.c_som._codebookptr()
        cdef view.array view = <float[:y,:x,:z]> data
        view[:,:,:] = arr.data

    @property
    def nrows(self):
        """\
        :returns: the number of rows in the network.
        :rtype: int
        """
        return self.c_som.gety()

    @property
    def ncols(self):
        """\
        :returns: the number of columns in the network.
        :rtype: int
        """
        return self.c_som.getx()

    @property
    def dim(self):
        """\
        :returns: the dimension of the input vectors.
        :rtype: int
        """
        return self.c_som.getz()


cdef class Som:
    """\
    Uses the SD-SOM algorithm (online learning).

    :param h: the network height
    :type h: int
    :param w: the network width
    :type w: int
    :param d: the dimension of input vectors
    :type d: int
    :param topol: the network topology
    :type topol: :const:`topology.RECT` or :const:`topology.HEXA`
    :param verbose: verbosity parameter
    :type verbose: bool
    """

    cdef _Som * c_som

    def __cinit__(self, size_t h, size_t w, size_t d, topology topol=topology.RECT, bool_t verbose=False):
        self.c_som = new _Som(h, w, d, topol, verbose)

    def __dealloc__(self):
        del self.c_som

    def train(self, data, tmax=None, r0=None, a0=0.5, rN=0.5, aN=1.e-10, std=0.3, rcool=cooling.LINEAR, acool=cooling.LINEAR):
        """\
        Train the network with data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.float32`)
        :type data: :class:`scipy.sparse.spmatrix`
        :param tmax: number of iterations
        :type tmax: int
        :param r0: radius at the first iteration
        :type r0: float
        :param a0: learning-rate at the first iteration
        :type a0: float
        :param rN: radius at the last iteration
        :type rN: float
        :param aN: learning-rate at the last iteration
        :type aN: float
        :param rcool: radius cooling strategy
        :type rcool: :const:`cooling.LINEAR` or :const:`cooling.EXPONENTIAL`
        :param acool: alpha cooling strategy
        :type acool: :const:`cooling.LINEAR` or :const:`cooling.EXPONENTIAL`
        """
        if r0 is None:
            r0 = min(self.c_som.gety(), self.c_som.getx()) / 2
        if tmax is None:
            tmax = 10 * data.shape[0]
        cdef vector[sparse_vec] m
        sparse_matrix_from_scipy(m, data)
        self.c_som.train(m, tmax, r0, a0, rN, aN, std, rcool, acool)

    def bmus(self, data):
        """\
        Return the best match units for data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.float32`)
        :type data: :class:`scipy.sparse.spmatrix`
        :returns: an array of the bmus coordinates (y,x)
        :rtype: 2D :class:`numpy.ndarray`
        """
        codebook = self.codebook
        codebook.shape = (self.ncols*self.nrows, self.dim)
        dst = -2 * data.dot(codebook.T)
        dst += (codebook ** 2).sum(axis=1)
        dst += data.power(2).sum(axis=1)
        bmus = np.argmin(dst, axis=1)
        YX = np.unravel_index(bmus, (self.nrows, self.ncols))
        return np.vstack(YX).transpose()

    @property
    def codebook(self):
        """\
        :returns: a view of the internal codebook.
        :rtype: 3D :class:`numpy.ndarray`
        """
        y, x, z = self.c_som.gety(), self.c_som.getx(), self.c_som.getz()
        cdef double * data = self.c_som._codebookptr()
        cdef view.array arr = <double[:y,:x,:z]> data
        return np.asarray(arr)

    @codebook.setter
    def codebook(self, arr):
        """\
        Set the internal codebook.

        :param arr: the new codebook
        :type arr: 3D :class:`numpy.ndarray`
        """
        y, x, z = self.c_som.gety(), self.c_som.getx(), self.c_som.getz()
        assert (y, x, z) == arr.shape
        if arr.dtype != np.float64:
            print('changing codebook type to float64')
            arr = arr.astype(np.float64)
        cdef double * data = self.c_som._codebookptr()
        cdef view.array view = <double[:y,:x,:z]> data
        view[:,:,:] = arr.data

    @property
    def nrows(self):
        """\
        :returns: the number of rows in the network.
        :rtype: int
        """
        return self.c_som.gety()

    @property
    def ncols(self):
        """\
        :returns: the number of columns in the network.
        :rtype: int
        """
        return self.c_som.getx()

    @property
    def dim(self):
        """\
        :returns: the dimension of the input vectors.
        :rtype: int
        """
        return self.c_som.getz()