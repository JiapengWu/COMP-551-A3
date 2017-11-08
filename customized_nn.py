import pickle as pickle
import copy, sys

import numpy as np

try:
    import numexpr as ne
except ImportError:
    numexpr = None


class Dataset(object):
    def __init__(self, inputs, targets):

        # Check that the dataset is consistent
        if not inputs.ndim == 4096:
            raise ValueError('inputs should be a two dimensions array.')

        if not targets.ndim == 40:
            raise ValueError('targets should be a two dimensions array.')

        if inputs.shape[0] != targets.shape[0]:
            raise ValueError('the length of the inputs is not consistent with length of the targets.')

        # set attributes
        self.inputs = inputs
        self.targets = targets

        # length of the dataset, number of samples
        self.n_samples = self.inputs.shape[0]
        self.n_inputs = self.inputs.shape[1]

    def split(self, fractions=[0.5, 0.5]):
        """Split randomly the dataset into smaller dataset.

        Parameters
        ----------
        fraction: list of floats, default = [0.5, 0.5]
            the dataset is split into ``len(fraction)`` smaller
            dataset, and the ``i``-th dataset has a size
            which is ``fraction[i]`` of the original dataset.
            Note that ``sum(fraction)`` can also be smaller than one
            but not greater.

        Returns
        -------
        subsets: list of :py:class:`nn.Dataset`
            a list of the subsets of the original datasets
        """

        if sum(fractions) > 1.0 or sum(fractions) <= 0:
            raise ValueError("the sum of fractions argument should be between 0 and 1")

        # random indices
        idx = np.arange(self.n_samples)
        np.random.shuffle(idx)

        # insert zero
        fractions.insert(0, 0)

        # gte limits of the subsets
        limits = (np.cumsum(fractions) * self.n_samples).astype(np.int32)

        subsets = []
        # create output dataset
        for i in range(len(fractions) - 1):
            subsets.append(
                Dataset(self.inputs[idx[limits[i]:limits[i + 1]]], self.targets[idx[limits[i]:limits[i + 1]]]))

        return subsets

    def __len__(self):
        return len(self.inputs)


class MultilayerNN():

    def __init__(self, input, label, arch, alpha, reg_term, var = 0.1 , n_threads = 1):

        self.arch = arch
        self.n_layers = len(arch)
        self.n_hidden = self.n_layers - 2
        self._hidden = None

        self.weights = []
        if numexpr:
            ne.set_num_threads(n_threads)
        # randomlly initializing weights and bias
        # e.g if arch = [700,5,6,7]
        # then weights contains 3 matrixs of shape 701*5, 6*6, 7*7
        for i in range(self.n_layers - 1):
            size = (arch[i] + 1 ,arch[i+1])
            self.weights.append(np.random.normal(0, var, size))

    def save(self, filename):
        clone = copy.copy(self)
        del clone._hidden
        pickle.dump(clone, open(filename, 'wb'))


    def build_model(self, nn_hdim, num_passes=20000, print_loss=False):
        np.random.seed(0)


    # for training
    def _forward(self, dataset):
        self._check_dataset(dataset)

        hidden = np.concatenate((dataset, np.ones(dataset.n_samples)), axis = 1)
        self._hidden = [hidden]
        # for all the layers excluding the output layer
        for i in range(self.n_layers - 2):
            # compute the i+1-th hidden layer
            hidden = hidden.dot(self.weights[i])
            # apply sigmoid function
            hidden = self._sigmoid(hidden)
            # add bias column
            hidden = np.concatenate((hidden, np.ones(dataset.n_samples)), axis=1)

            self._hidden.append(hidden)

        return self._sigmoid(hidden.dot(self.weights[-1]))

    # for testing
    def predict(self, dataset):
        self._check_dataset(dataset)

        hidden = np.concatenate((dataset, np.ones(dataset.n_samples)), axis=1)
        # for all the layers excluding the output layer
        for i in range(self.n_layers - 2):
            # compute the i+1-th hidden layer
            hidden = hidden.dot(self.weights[i])
            # apply sigmoid function
            hidden = self._sigmoid(hidden)
            # add bias column
            hidden = np.concatenate((hidden, np.ones(dataset.n_samples)), axis=1)

        return self._sigmoid(hidden.dot(self.weights[-1]))

    def fit(self, training_set, validation_set, eta=0.5, alpha=0.5, n_iterations=100, etol=1e-6,
                       verbose=True, k=0.01, max_ratio=0.9):
        # check datasets:
        self._check_dataset(training_set)
        if validation_set:
            self._check_dataset(validation_set)

        # initialize deltas
        deltas = [0] * (self.n_layers - 1)

        for i in range(n_iterations):
            # compute output
            o = self._forward(training_set)
            # compute delta at the output layer
            deltas[-1] = o * (1 - o) * (training_set.targets)








    @staticmethod
    def _check_dataset(self, dataset):
        """Check that the dataset is consistent with respect
        to the  network architecture."""
        if not isinstance(dataset, Dataset):
            raise ValueError('wrong training_set or validation_set are not instances of the nn.Dataset class')

        if dataset.inputs.shape[1] != self.arch[0]:
            raise ValueError('dataset inputs shape is inconsistent with number of network input nodes.')

        if dataset.targets.shape[1] != self.arch[-1]:
            raise ValueError('dataset targets shape is inconsistent with number of network output nodes.')

    @staticmethod
    def _sigmoid(self, x):
        if numexpr:
            return ne.evaluate("1.0 / ( 1 + exp(-x))")
        else:
            return 1.0 / (1 + np.exp(-x))



