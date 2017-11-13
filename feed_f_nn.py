__author__="qianyuan"

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
        if not inputs.shape[1] == 4096:
            raise ValueError('inputs should be a 4096 dimensions array.')

        if not targets.shape[1] == 40:
            raise ValueError('targets should be a 40 dimensions array.')

        if inputs.shape[0] != targets.shape[0]:
            raise ValueError('the length of the inputs is not consistent with length of the targets.')

        # set attributes
        self.inputs = inputs
        self.targets = targets

        # length of the dataset, number of samples
        self.n_samples = self.inputs.shape[0]
        self.n_features = self.inputs.shape[1]

    def split(self, fractions=[0.8, 0.2]):
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

    def __init__(self, arch, beta=1, var = 0.1, n_threads = 1):

        """
        :param arch: list of integers
        :param beta: steepness of the sigmoidal function in 'x=0'
        :param var: the variance of the weights
        :param n_threads: the number of threads
        """

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


    # for training
    def _forward(self, dataset):
        """compute network output"""

        self._check_dataset(dataset)

        hidden = np.column_stack([dataset.inputs, np.ones((dataset.n_samples, 1))])
        self._hidden = [hidden]
        # for all the layers excluding the output layer
        for i in range(self.n_layers - 2):
            # compute the i+1-th hidden layer
            hidden = np.dot(hidden, self.weights[i])
            # apply sigmoid function
            hidden = self._sigmoid(hidden)
            # add bias column
            hidden = np.column_stack([hidden, -np.ones((dataset.n_samples, 1))])
            self._hidden.append(hidden)

        # using softmax on last layer
        output = np.dot(hidden, self.weights[-1])
        output= self._softmax(output)

        return output


    # for testing
    def predict(self, dataset):
        hidden = np.column_stack([dataset, np.ones((dataset.shape[0], 1))])
        # for all the layers excluding the output layer
        for i in range(self.n_layers - 2):
            # compute the i+1-th hidden layer
            hidden = np.dot(hidden, self.weights[i])
            # apply sigmoid function
            hidden = self._sigmoid(hidden)
            # add bias column
            hidden = np.column_stack([hidden, -np.ones((dataset.shape[0], 1))])

        # using softmax on last layer
        output = np.dot(hidden, self.weights[-1])
        output = self._softmax(output)
        output = output.argmax(axis=1)
        print output
        return output


    def _predict(self, dataset):
        self._check_dataset(dataset)
        hidden = np.column_stack([dataset.inputs, np.ones((dataset.n_samples, 1))])
        # for all the layers excluding the output layer
        for i in range(self.n_layers - 2):
            # compute the i+1-th hidden layer
            hidden = np.dot(hidden, self.weights[i])
            # apply sigmoid function
            hidden = self._sigmoid(hidden)
            # add bias column
            hidden = np.column_stack([hidden, -np.ones((dataset.n_samples, 1))])

        # using softmax on last layer
        output = np.dot(hidden, self.weights[-1])
        output = self._softmax(output)
        return output


    def backprop(self, training_set, validation_set, eta=0.5, alpha=0.5, n_iterations=100, etol=1e-10,
                       verbose=True, k=0.01, max_ratio=0.9):
        """train the network using backpropagation

            eta: initial learning rate
            alpha: momentum term
            n_iterations:the number of epochs
            etol: threshold of the error
            k: define how to change learning rate
        """
        # check datasets:
        self._check_dataset(training_set)
        if validation_set:
            self._check_dataset(validation_set)

        # initialize deltas
        deltas = [0] * (self.n_layers - 1)

        #save errors at each iteration on both training and validation set
        train_err = np.zeros(n_iterations + 1)
        train_err[0] = self.error(training_set)
        if validation_set:
            val_err = np.zeros(n_iterations +1)
            val_err[0] = self.error(validation_set)

        #initialize weights at previous step
        pre_weight = [np.zeros_like(w) for w in self.weights]

        #update the value of weights
        for n in range(n_iterations):
            o = self._forward(training_set)
            deltas[-1] = (o - training_set.targets)

            #for each hidden layer, calculate deltas
            for i in range(self.n_hidden):
                j = -(i+1)
                deltas[j - 1] = self._hidden[j][:, :-1] * (1.0 - self._hidden[j][:, :-1]) * \
                                np.dot(deltas[j],self.weights[j][:-1].T)

            #update weights
            for i in range(self.n_layers - 1):
                pre_weight[i] = alpha * pre_weight[i] + eta * \
                                                        np.dot(deltas[i].T, self._hidden[i]).T / training_set.n_samples
                self.weights[i] -= pre_weight[i]

            # save error
            train_err[n] = self.error(training_set)
            if validation_set:
                val_err[n] = self.error(validation_set)
                print

            print "{}-th epoch, training error: {}, validation error: {}".format(
                                                        str(n), str(train_err[n]), str(val_err[n]))
            #stop training when we are close to minimum
            if np.abs(train_err[n] - train_err[n-1]) < etol:
                print("Already got the minumum error")
                break

            # check error behaviour and change learning parameter
            if train_err[n] > train_err[n-1]:
                eta /= 1 + k
            else:
                eta *= 1 + float(k) / 10

        return train_err

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
    def _sigmoid(x):
        if numexpr:
            return ne.evaluate("1.0 / ( 1 + exp(-x))")
        else:
            return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def _softmax(z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]  # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]  # dito
        return e_x / div

    def error(self, dataset):
        prediction = self._predict(dataset)
        print "Accuracy: " + str(np.mean(prediction.argmax(axis=1) == dataset.targets.argmax(axis=1)))
        return np.mean((prediction - dataset.targets) ** 2)
