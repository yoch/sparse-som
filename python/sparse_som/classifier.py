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
        self.quant_error = None
        self.topog_error = None

    def fit(self, data, labels, **kwargs):
        """\
        Training the SOM on the the data and calibrate itself.

        After the training, `self.quant_error` and `self.topog_error` are 
        respectively set.

        :param data: sparse input matrix (ideal dtype is `numpy.float32`)
        :type data: :class:`scipy.sparse.csr_matrix`
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
        """\
        Calibrate the network using `self._bmus`.
        """
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

        :param data: sparse input matrix (ideal dtype is `numpy.float32`)
        :type data: :class:`scipy.sparse.csr_matrix`
        :returns: the BMUs, the QE and the TE
        :rtype: tuple
        """
        bmus, seconds, mdst = self._som._bmus_and_seconds(data)
        quant_error = np.sqrt(mdst, out=mdst).mean()
        topog_error = self._som._topographic_error(bmus, seconds, data.shape[0])
        return self._som._to_bmus(bmus), quant_error, topog_error

    def _predict_from_bmus(self, bmus, unkown):
        lst = []
        for i,j in bmus:
            cls = self.classifier.get((i,j))
            if cls is None:
                lst.append(unkown)
            else:
                lbl, p = cls
                lst.append(lbl)
        return np.array(lst)
        
    def predict(self, data, unkown=None):
        """\
        Classify data according to previous calibration.

        :param data: sparse input matrix (ideal dtype is `numpy.float32`)
        :type data: :class:`scipy.sparse.csr_matrix`
        :param unkown: the label to attribute if no label is known
        :returns: the labels guessed for data
        :rtype: `numpy.array`
        """
        assert self.classifier is not None, 'not calibrated'
        bmus = self._som.bmus(data)
        return self._predict_from_bmus(bmus, unkown)

    def fit_predict(self, data, labels, unkown=None):
        """\
        Fit and classify data efficiently.

        :param data: sparse input matrix (ideal dtype is `numpy.float32`)
        :type data: :class:`scipy.sparse.csr_matrix`
        :param labels: the labels associated with data
        :type labels: iterable
        :param unkown: the label to attribute if no label is known
        :returns: the labels guessed for data
        :rtype: `numpy.array`
        """
        self.fit(data, labels)
        return self._predict_from_bmus(self._bmus, unkown)

    def get_precision(self):
        """\
        :returns: the ratio part of the dominant label for each unit.
        :rtype: 2D :class:`numpy.ndarray`
        """
        assert self.classifier is not None, 'not calibrated'
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
            assert self._bmus is not None, 'not trained'
            bmus = self._bmus
        arr = np.zeros((self._som.nrows, self._som.ncols))
        for i,j in bmus:
            arr[i,j] += 1
        return arr

    def __repr__(self):
        return '<SomClassifier: %dx%d units>' % (self._som.nrows, self._som.ncols)
