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

    def fit(self, data, labels, **kwargs):
        """\
        Training the SOM on the the data and calibrate itself.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.float32`)
        :type data: :class:`scipy.sparse.spmatrix`
        :param labels: the labels associated with data
        :type labels: iterable
        :param \**kwargs: optional parameters for :meth:`train`
        """
        self._som.train(data, **kwargs)
        self._calibrate(data, labels)

    def _calibrate(self, data, labels, _bmus=None):
        if _bmus is None:
            _bmus = self.bmus(data)
            print(_bmus)
        # network calibration
        classifier = defaultdict(Counter)
        for (i,j), label in zip(_bmus, labels):
            classifier[i,j][label] += 1
        self.classifier = {}
        for ij, cnt in classifier.items():
            maxi = max(cnt.items(), key=itemgetter(1))
            nb = sum(cnt.values())
            self.classifier[ij] = maxi[0], maxi[1] / nb
        return _bmus

    def error_metrics(self, data):
        """\
        Compute common error metrics (Quantization err. and Topographic err.)
        for this data.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.float32`)
        :type data: :class:`scipy.sparse.spmatrix`
        :returns: the BMUs, the QE and the TE
        :rtype: tuple
        """
        assert isinstance(self._som, som.BSom), 'Error metrics must use BSom classifier'
        bmus, seconds, mdst, sdst = self._som._bmus_and_seconds(data)
        # correct min dist, because we use external CSR without _sqsum
        mdst += data.power(2).sum(axis=1).A1
        np.clip(mdst, 0, None, mdst)
        np.sqrt(mdst, mdst)
        quantization_err = mdst.mean()
        # BEWARE: rectangular dist only...
        shape = (self._som.nrows, self._som.ncols)
        #y1, x1 = np.unravel_index(bmus, shape)
        #y2, x2 = np.unravel_index(seconds, shape)
        #dy = abs(y1 - y2)
        #dx = abs(x1 - x2)
        #topographic_err = (np.maximum(dy, dx) > 1).mean()
        topographic_err = self._som._topographic_error(bmus, seconds, data.shape[0])
        YX = np.unravel_index(bmus, shape)
        return np.vstack(YX).T, quantization_err, topographic_err

    def predict(self, data, unkown=None, _bmus=None):
        """\
        Classify data according to previous calibration.

        :param data: sparse input matrix (ideally :class:`csr_matrix` of `numpy.float32`)
        :type data: :class:`scipy.sparse.spmatrix`
        :param unkown: the label to attribute if no label is known
        :returns: the labels guessed for data
        :rtype: list
        """
        if not hasattr(self, 'classifier'):
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
        return lst

    def get_precision(self):
        """\
        :returns: the ratio part of the dominant label for each unit.
        :rtype: 2D :class:`numpy.ndarray`
        """
        if not hasattr(self, 'classifier'):
            raise RuntimeError('not calibrated')
        arr = np.zeros((self._som.nrows, self._som.ncols))
        for ij, (lbl, p) in self.classifier.items():
            arr[ij] = p
        return arr

    def histogram(self, bmus):
        """\
        Return a 2D histogram of bmus.

        :param bmus: the best-match units indexes for underlying data.
        :type bmus: :class:`numpy.ndarray`
        :returns: the computed 2D histogram of bmus.
        :rtype: :class:`numpy.ndarray`
        """
        arr = np.zeros((self._som.nrows, self._som.ncols))
        for i,j in bmus:
            arr[i,j] += 1
        return arr

    def __repr__(self):
        return '<SomClassifier: %dx%d units>' % (self._som.nrows, self._som.ncols)
