# distutils: language = c++
# distutils: sources = lib/bsom.cpp, lib/som.cpp


cimport cython
cimport numpy as np
from cython cimport view
import numpy as np
import scipy.sparse



cdef extern from "lib/data.h":
    cdef cppclass CSR:
        float * data
        int * indices
        int * indptr
        float * _sqsum
        int nrows
        int ncols
        int nnz


cdef CSR csrmat_from_spsparse(sm):
    "convert scipy.sparse.spmatrix to internal csr wrapper"   
    # ensure that sm is in the correct format
    ensure_validity(sm)
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


def ensure_validity(sm):
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
        void getBmus(const CSR&, size_t *, float *, size_t *, float *)
        double topographicError(size_t * const bmus, size_t * const second, size_t n)
        size_t getx()
        size_t gety()
        size_t getz()
        float * _codebookptr()
        int getverb()
        void setverb(int)

    cdef cppclass _Som "som::Som":
        _Som(size_t, size_t, size_t, topology, int)
        void train(const CSR&, size_t, double, double, double, double, double, cooling, cooling)
        size_t getx()
        size_t gety()
        size_t getz()
        double * _codebookptr()
        int getverb()
        void setverb(int)


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


    def bmus(self, data):
        """\
        Return the best match units for data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.single`)
        :type data: :class:`scipy.sparse.spmatrix`
        :returns: an array of the bmus coordinates (y,x)
        :rtype: 2D :class:`numpy.ndarray`
        """
        cdef CSR m = csrmat_from_spsparse(data)
        cdef np.ndarray[size_t, ndim=1] bmus = np.empty(m.nrows, dtype=np.uintp)
        cdef np.ndarray[float, ndim=1] mdst = np.empty(m.nrows, dtype=np.single)
        self.c_som.getBmus(m, <size_t*> bmus.data, <float*> mdst.data, NULL, NULL)
        YX = np.unravel_index(bmus, (self.nrows, self.ncols))
        return np.vstack(YX).transpose()

    def _bmus_and_seconds(self, data):
        """\
        Return the best match units for data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.single`)
        :type data: :class:`scipy.sparse.spmatrix`
        :returns: an array of the bmus coordinates (y,x)
        :rtype: 2D :class:`numpy.ndarray`
        """
        cdef CSR m = csrmat_from_spsparse(data)
        cdef np.ndarray[size_t, ndim=1] bmus = np.empty(m.nrows, dtype=np.uintp)
        cdef np.ndarray[float, ndim=1] mdst = np.empty(m.nrows, dtype=np.single)
        cdef np.ndarray[size_t, ndim=1] seconds = np.empty(m.nrows, dtype=np.uintp)
        cdef np.ndarray[float, ndim=1] sdst = np.empty(m.nrows, dtype=np.single)
        self.c_som.getBmus(m, <size_t*> bmus.data, <float*> mdst.data, <size_t*> seconds.data, <float*> sdst.data)
        return bmus, seconds, mdst, sdst        

    def _topographic_error(self, np.ndarray[size_t, ndim=1] bmus, np.ndarray[size_t, ndim=1] seconds, int nsamples):
        return self.c_som.topographicError(<size_t*> bmus.data, <size_t*> seconds.data, nsamples)

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
        self.c_som.train(m, tmax, r0, a0, rN, aN, std, rcool, acool)

    def _dst_argmin_min(self, data):
        codebook = self.codebook
        shape = codebook.shape
        codebook.shape = (-1, self.dim)
        dst = -2 * data.dot(codebook.T)
        dst += (codebook ** 2).sum(axis=1)
        dst += data.power(2).sum(axis=1)
        codebook.shape = shape
        bmus = dst.argmin(axis=1)
        mdst = dst.min(axis=1)
        np.clip(mdst, 0, None, mdst)
        np.sqrt(mdst, mdst)
        return bmus, mdst

    def bmus(self, data):
        """\
        Return the best match units for data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.single`)
        :type data: :class:`scipy.sparse.spmatrix`
        :returns: an array of the bmus coordinates (y,x)
        :rtype: 2D :class:`numpy.ndarray`
        """
        bmus, _ = self._dst_argmin_min(data)
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
        cdef view.array vdata = <double[:y,:x,:z]> data
        vdata[:,:,:] = arr.data

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
