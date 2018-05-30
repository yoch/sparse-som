from . import som
from collections import defaultdict, Counter
from operator import itemgetter
import numpy as np


class SomClassifier:
    def __init__(self, cls=som.BSom, *args, **kwargs):
        """\
        :param cls: SOM constructor
        :type cls: :class:`Som` or :class:`BSom`
        :param \*args: positional parameters for the constructor
        :param \**kwargs: named parameters for the constructor
        """
        self._som = cls(*args, **kwargs)
        self._bmus = None   # bmus of the training data
        self.classifier = None
        self.qerror = None
        self.terror = None

    def fit(self, data, labels, **kwargs):
        """\
        Training the SOM on the the data and calibrate itself.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.float32`)
        :type data: :class:`scipy.sparse.spmatrix`
        :param labels: the labels associated with data
        :type labels: iterable
        :param \**kwargs: optional parameters for :meth:`train`
        """
        # train the network
        self._som.train(data, **kwargs)
        # retrieve first and second bmus and distances
        bmus, q_error, t_error = self.bmus_with_errors(data)
        # set errors measures of training data
        self.quant_error = q_error
        self.topog_error = t_error
        # store training bmus
        self._bmus = bmus
        # calibrate
        self._calibrate(data, labels)

    def _calibrate(self, data, labels):
        # network calibration
        classifier = defaultdict(Counter)
        for (i,j), label in zip(self._bmus, labels):
            classifier[i,j][label] += 1
        self.classifier = {}
        for ij, cnt in classifier.items():
            maxi = max(cnt.items(), key=itemgetter(1))
            nb = sum(cnt.values())
            self.classifier[ij] = maxi[0], maxi[1] / nb

    def bmus_with_errors(self, data):
        """\
        Compute common error metrics (Quantization err. and Topographic err.)
        for this data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.float32`)
        :type data: :class:`scipy.sparse.spmatrix`
        :returns: the BMUs, the QE and the TE
        :rtype: tuple
        """
        bmus, seconds, mdst = self._som._bmus_and_seconds(data)
        quant_error = np.sqrt(mdst, out=mdst).mean()
        topog_error = self._som._topographic_error(bmus, seconds, data.shape[0])
        return self._som._to_bmus(bmus), quant_error, topog_error

    def predict(self, data, unkown=None, _bmus=None):
        """\
        Classify data according to previous calibration.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.float32`)
        :type data: :class:`scipy.sparse.spmatrix`
        :param unkown: the label to attribute if no label is known
        :returns: the labels guessed for data
        :rtype: list
        """
        if self.classifier is None:
            raise RuntimeError('not calibrated')
        if _bmus is None:
            _bmus = self._som.bmus(data)
        lst = []
        for i,j in _bmus:
            cls = self.classifier.get((i,j))
            if cls is None:
                lst.append(unkown)
            else:
                lbl, p = cls
                lst.append(lbl)
        return np.array(lst)

    def get_precision(self):
        """\
        :returns: the ratio part of the dominant label for each unit.
        :rtype: 2D :class:`numpy.ndarray`
        """
        if self.classifier is None:
            raise RuntimeError('not calibrated')
        arr = np.zeros((self._som.nrows, self._som.ncols))
        for ij, (lbl, p) in self.classifier.items():
            arr[ij] = p
        return arr

    def histogram(self, bmus=None):
        """\
        Return a 2D histogram of bmus.

        :param bmus: the best-match units indexes for underlying data.
        :type bmus: :class:`numpy.ndarray`
        :returns: the computed 2D histogram of bmus.
        :rtype: :class:`numpy.ndarray`
        """
        if bmus is None:
            bmus = self._bmus
        arr = np.zeros((self._som.nrows, self._som.ncols))
        for i,j in bmus:
            arr[i,j] += 1
        return arr

    def __repr__(self):
        return '<SomClassifier: %dx%d units>' % (self._som.nrows, self._som.ncols)
