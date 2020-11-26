from . import som
from collections import defaultdict, Counter
from operator import itemgetter
import numpy as np


class SomClassifier:
    def __init__(self, cls=som.BSom, height=10, width=10, dim=None, **kwargs):
        """\
        :param cls: SOM constructor
        :type cls: :class:`Som` or :class:`BSom`
        :param height: SOM height (default 10)
        :type height: int
        :param width: SOM width (default 10)
        :type width: int
        :param dim: SOM nodes nb of dimensions (if None, detected when :class:`fit` called)
        :type dim: int
        :param \**kwargs: named parameters for the constructor
        """
        # take any positional arguments to named
        kwargs['h'] = height
        kwargs['w'] = width
        if dim is not None:
            kwargs['d'] = dim

        self._cls = cls             # the network constructor
        self._kwargs = kwargs       # named params for constructor
        self._som = None            # the SOM network
        self._bmus = None           # bmus of the training data
        self.classifier = None      # used for network calibration
        self.quant_error = None
        self.topog_error = None

    def setup(self, **kwargs):
        """\
        Add / Change SOM constructor parameters.
        """
        assert self._som is None, 'cannot setup params after SOM instanciation'
        self._kwargs.update(kwargs)

    def params(self):
        """\
        Get the SOM constructor parameters.
        """
        return self._kwargs

    @property
    def som(self):
        # lazy constructor
        if self._som is None:
            self._som = self._cls(**self._kwargs)
        return self._som

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

        # set vectors dimensions if they're missing
        if 'd' not in self._kwargs:
            self._kwargs['d'] = data.shape[1]

        assert self.som.dim == data.shape[1], 'dimension mismatch'

        # train the network
        self.som.train(data, **kwargs)
        # retrieve first and second bmus and distances
        bmus = self.som.bmus(data, True, True)
        # set errors measures of training data
        self.quant_error = self.som.quantization_error
        self.topog_error = self.som.topographic_error
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
        for (i, j), label in zip(self._bmus, labels):
            classifier[i, j][label] += 1
        self.classifier = {}
        for ij, cnt in classifier.items():
            maxi = max(cnt.items(), key=itemgetter(1))
            nb = sum(cnt.values())
            self.classifier[ij] = maxi[0], maxi[1] / nb

    def _predict_from_bmus(self, bmus, unkown):
        lst = []
        for i, j in bmus:
            cls = self.classifier.get((i, j))
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
        bmus = self.som.bmus(data)
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
        arr = np.zeros((self.som.nrows, self.som.ncols))
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
        arr = np.zeros((self.som.nrows, self.som.ncols))
        for i, j in bmus:
            arr[i, j] += 1
        return arr

    def __repr__(self):
        return '<SomClassifier: ' + ', '.join(map('%s=%s'.__mod__, self._kwargs.items())) + '>'
