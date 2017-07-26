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

    def _calibrate(self, data, labels):
        classifier = defaultdict(Counter)
        bmus = self._som.bmus(data)
        for (i,j), label in zip(bmus, labels):
            classifier[i,j][label] += 1
        self.classifier = {}
        for ij, cnt in classifier.items():
            maxi = max(cnt.items(), key=itemgetter(1))
            nb = sum(cnt.values())
            self.classifier[ij] = maxi[0], maxi[1] / nb

    def predict(self, data, unkown=None):
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
        lst = []
        for i,j in self._som.bmus(data):
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
