# distutils: language = c++
# distutils: sources = lib/bsom.cpp, lib/som.cpp, lib/data.cpp
# cython: language_level=3, boundscheck=False

cimport cython
cimport numpy as np
from cython cimport view
from libcpp cimport bool
import numpy as np
import scipy.sparse

try:
    from sklearn.metrics.pairwise import pairwise_distances
except ImportError:
    HAS_SKLEARN = False

    def eucdist(codebook, data):
        dst = -2 * data.dot(codebook.T)
        dst += (codebook ** 2).sum(axis=1)
        dst += data.power(2).sum(axis=1)
        np.clip(dst, 0, None, out=dst)
        np.sqrt(dst, out=dst)
        return dst
else:
    HAS_SKLEARN = True


cdef extern from "lib/data.h":
    cdef cppclass CSR:
        CSR()
        #CSR(float*, int*, int*, int, int, int)
        void initSqSum()
        float * data
        int * indices
        int * indptr
        int nrows
        int ncols
        int nnz
        float * _sqsum


cdef CSR csrmat_from_spsparse(sm):
    "convert scipy.sparse.spmatrix to internal csr wrapper"
    # ensure that sm is in the correct format
    _ensure_validity(sm)
    # retrieve refs on internal array
    cdef np.ndarray[float, ndim=1] data = sm.data
    cdef np.ndarray[int, ndim=1] indices = sm.indices
    cdef np.ndarray[int, ndim=1] indptr = sm.indptr
    # convert to internal representation
    cdef CSR csr
    csr.data = <float*> data.data
    csr.indices = <int*> indices.data
    csr.indptr = <int*> indptr.data
    csr.nrows = sm.shape[0]
    csr.ncols = sm.shape[1]
    csr.nnz = sm.nnz
    return csr


def _ensure_validity(sm):
    # sanity checks
    assert scipy.sparse.isspmatrix_csr(sm), "data must be at CSR format"
    #if not scipy.sparse.isspmatrix(sm):
    #    print("convert to CSR")
    #    sm = csr_matrix(sm, dtype=np.single)
    #if not scipy.sparse.isspmatrix_csr(sm):
    #    print("convert to CSR")
    #    sm = sm.tocsr()
    if not sm.dtype == np.single:
        #HACK: modify sm in place,
        #      (works but this is undocumented)
        print("convert %s to float" % sm.dtype)
        sm.data = sm.data.astype(np.single)
    if not sm.has_sorted_indices:
        print("sorting indices")
        sm.sort_indices()
    return sm


def _umatrix(som):
    # NOTE this is not implemented for the HEXA case
    H, W = som.nrows, som.ncols
    cb = som.codebook
    umat = np.zeros(shape=(H, W))
    for i in range(H):
        for j in range(W):
            v = cb[i, j]
            dist_sum = 0.0; ct = 0
            if i > 0:
                dist_sum += np.linalg.norm(v - cb[i-1, j]); ct += 1
            if i+1 < H:
                dist_sum += np.linalg.norm(v - cb[i+1, j]); ct += 1
            if j > 0:
                dist_sum += np.linalg.norm(v - cb[i, j-1]); ct += 1
            if j+1 < W:
                dist_sum += np.linalg.norm(v - cb[i, j+1]); ct += 1
            umat[i][j] = dist_sum / ct
    return umat


def _activation_map(codebook, data):
    if not HAS_SKLEARN:
        return eucdist(codebook, data)
    shape = codebook.shape
    codebook.shape = -1, shape[-1]
    #dst = euclidean_distances(codebook, data)
    dst = pairwise_distances(codebook, data, metric='euclidean', njobs=-1)
    codebook.shape = shape
    return dst


cdef extern from "lib/som.h" namespace "som":
    cpdef enum cooling:
        LINEAR = 0
        EXPONENTIAL = 1

    cpdef enum topology:
        RECT = 8
        HEXA = 6
        CIRC = 4

    cdef cppclass _BSom "som::BSom":
        _BSom(size_t, size_t, size_t, topology, int)
        void train(const CSR&, size_t, float, float, float, cooling)
        void getBmus(const CSR&, size_t *, float *, size_t *, bool)
        double topographicError(size_t * const bmus, size_t * const second, size_t n)
        size_t getx()
        size_t gety()
        size_t getz()
        float * _codebookptr()
        int getverb()
        void setverb(int)
        topology getTopology()

    cdef cppclass _Som "som::Som":
        _Som(size_t, size_t, size_t, topology, int)
        void train(const CSR&, size_t, double, double, double, double, double, cooling, cooling)
        void getBmus(const CSR&, size_t *, double *, size_t *, bool)
        double topographicError(size_t * const bmus, size_t * const second, size_t n)
        size_t getx()
        size_t gety()
        size_t getz()
        double * _codebookptr()
        int getverb()
        void setverb(int)
        topology getTopology()


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
    :type verbose: int (0..2)
    """

    cdef _BSom * c_som
    cdef dict __dict__

    def __cinit__(self, size_t h, size_t w, size_t d, topology topol=topology.RECT, int verbose=0):
        self.c_som = new _BSom(h, w, d, topol, verbose)

    def __dealloc__(self):
        del self.c_som

    def train(self, data, size_t epochs=10, float r0=0, float rN=0.5, float std=0.3, cooling cool=cooling.LINEAR):
        """\
        Train the network with data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.single`)
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
        cdef CSR m = csrmat_from_spsparse(data)
        self.c_som.train(m, epochs, r0, rN, std, cool)

    def _to_bmus(self, bmus):
        YX = np.unravel_index(bmus, (self.nrows, self.ncols))
        return np.vstack(YX).transpose()

    def bmus(self, data, tg_error=False, qt_error=False):
        """\
        Return the best match units for data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.single`)
        :type data: :class:`scipy.sparse.spmatrix`
        :param tg_error: indicates whether to calculate the topographic error (default: `False`)
        :type tg_error: bool
        :param qt_error: indicates whether to calculate the quantization error (default: `False`)
        :type qt_error: bool
        :returns: an array of the bmus coordinates (y,x)
        :rtype: 2D :class:`numpy.ndarray`
        """
        bmus, seconds, mdst = self._bmus_and_seconds(data)
        if tg_error:
            self.topographic_error = self._topographic_error(bmus, seconds, data.shape[0])
        else:
            self.topographic_error = None
        if qt_error:
            self.quantization_error = np.sqrt(mdst, out=mdst).mean()
        else:
            self.quantization_error = None
        return self._to_bmus(bmus)

    def _bmus_and_seconds(self, data):
        cdef CSR m = csrmat_from_spsparse(data)
        # important: initialize X^2 because we want correct mdst as result
        m.initSqSum()
        cdef np.ndarray[size_t, ndim=1] bmus = np.empty(m.nrows, dtype=np.uintp)
        cdef np.ndarray[float, ndim=1] mdst = np.empty(m.nrows, dtype=np.single)
        cdef np.ndarray[size_t, ndim=1] seconds = np.empty(m.nrows, dtype=np.uintp)
        #cdef np.ndarray[float, ndim=1] sdst = np.empty(m.nrows, dtype=np.single)
        self.c_som.getBmus(m, <size_t*> bmus.data, <float*> mdst.data, <size_t*> seconds.data, True)
        return bmus, seconds, mdst

    def _topographic_error(self, np.ndarray[size_t, ndim=1] bmus, np.ndarray[size_t, ndim=1] seconds, int nsamples):
        return self.c_som.topographicError(<size_t*> bmus.data, <size_t*> seconds.data, nsamples)

    def activation_map(self, data):
        """\
        Return the distance between each data sample and each codebook unit.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.single`)
        :type data: :class:`scipy.sparse.spmatrix`
        :returns: an array with shape (nsample, nunits)
        :rtype: 2D :class:`numpy.ndarray`
        """
        return _activation_map(self.codebook, data)

    @property
    def topol(self):
        return self.c_som.getTopology()

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
        if arr.dtype != np.single:
            print('changing codebook type to float')
            arr = arr.astype(np.single)
        cdef float* data = self.c_som._codebookptr()
        cdef view.array view = <float[:y,:x,:z]> data
        view[:,:,:] = arr.data

    @property
    def umatrix(self):
        """\
        :returns: the network U-matrix.
        :rtype: 2D :class:`numpy.ndarray`
        """
        return _umatrix(self)

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

    @property
    def verbose(self):
        return self.c_som.getverb()

    @verbose.setter
    def verbose(self, int v):
        self.c_som.setverb(v)


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
    :type verbose: int (0..2)
    """

    cdef _Som * c_som
    cdef dict __dict__

    def __cinit__(self, size_t h, size_t w, size_t d, topology topol=topology.RECT, int verbose=0):
        self.c_som = new _Som(h, w, d, topol, verbose)

    def __dealloc__(self):
        del self.c_som

    def train(self, data, tmax=None, r0=None, a0=0.5, rN=0.5, aN=1.e-10, std=0.3, rcool=cooling.LINEAR, acool=cooling.LINEAR):
        """\
        Train the network with data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.single`)
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
        cdef CSR m = csrmat_from_spsparse(data)
        # important: initialize X^2
        m.initSqSum()
        self.c_som.train(m, tmax, r0, a0, rN, aN, std, rcool, acool)


    def _to_bmus(self, bmus):
        YX = np.unravel_index(bmus, (self.nrows, self.ncols))
        return np.vstack(YX).transpose()

    def bmus(self, data, tg_error=False, qt_error=False):
        """\
        Return the best match units for data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.single`)
        :type data: :class:`scipy.sparse.spmatrix`
        :param tg_error: indicates whether to calculate the topographic error (default: `False`)
        :type tg_error: bool
        :param qt_error: indicates whether to calculate the quantization error (default: `False`)
        :type qt_error: bool
        :returns: an array of the bmus coordinates (y,x)
        :rtype: 2D :class:`numpy.ndarray`
        """
        bmus, seconds, mdst = self._bmus_and_seconds(data)
        if tg_error:
            self.topographic_error = self._topographic_error(bmus, seconds, data.shape[0])
        else:
            self.topographic_error = None
        if qt_error:
            self.quantization_error = np.sqrt(mdst, out=mdst).mean()
        else:
            self.quantization_error = None
        return self._to_bmus(bmus)

    def _bmus_and_seconds(self, data):
        cdef CSR m = csrmat_from_spsparse(data)
        # important: initialize X^2 because we want correct mdst as result
        m.initSqSum()
        cdef np.ndarray[size_t, ndim=1] bmus = np.empty(m.nrows, dtype=np.uintp)
        cdef np.ndarray[double, ndim=1] mdst = np.empty(m.nrows, dtype=np.double)
        cdef np.ndarray[size_t, ndim=1] seconds = np.empty(m.nrows, dtype=np.uintp)
        #cdef np.ndarray[double, ndim=1] sdst = np.empty(m.nrows, dtype=np.double)
        self.c_som.getBmus(m, <size_t*> bmus.data, <double*> mdst.data, <size_t*> seconds.data, True)
        return bmus, seconds, mdst

    def _topographic_error(self, np.ndarray[size_t, ndim=1] bmus, np.ndarray[size_t, ndim=1] seconds, int nsamples):
        return self.c_som.topographicError(<size_t*> bmus.data, <size_t*> seconds.data, nsamples)

    def activation_map(self, data):
        """\
        Return the distance between each data sample and each codebook unit.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.single`)
        :type data: :class:`scipy.sparse.spmatrix`
        :returns: an array with shape (nsample, nunits)
        :rtype: 2D :class:`numpy.ndarray`
        """
        return _activation_map(self.codebook, data)

    @property
    def topol(self):
        return self.c_som.getTopology()

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
        cdef view.array vdata = <double[:y,:x,:z]> data
        vdata[:,:,:] = arr.data

    @property
    def umatrix(self):
        """\
        :returns: the network U-matrix.
        :rtype: 2D :class:`numpy.ndarray`
        """
        return _umatrix(self)

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

    @property
    def verbose(self):
        return self.c_som.getverb()

    @verbose.setter
    def verbose(self, int v):
        self.c_som.setverb(v)
