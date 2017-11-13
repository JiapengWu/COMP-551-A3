from feed_f_nn import Dataset, MultilayerNN
import numpy as np
from sklearn import preprocessing
from keras.utils import np_utils
import csv

def load_x():
	path = "./data/training_data.npy"
	training_data_x = np.load(path)
	training_data_x = training_data_x.reshape(50000, 64*64)

	print("Training data x: {}".format(training_data_x.shape))

	return training_data_x


def load_y():
	path = "./data/train_y.csv"
	training_data_y = np.loadtxt(path, delimiter="\n")
	training_data_y = np.loadtxt(path, delimiter="\n")
	# Label Encode our results.
	# [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.
	# 15.  16.  17.  18.  20.  21.  24.  25.  27.  28.  30.  32.  35.  36.  40.
	# 42.  45.  48.  49.  54.  56.  63.  64.  72.  81.]
	# will be encoded to 0,1,2,3,...
	# this means that "81" will be [0,0,0,0,0,...,0,1] where the 40th entry is 1
	le = preprocessing.LabelEncoder()

	le.fit(training_data_y)  # fit with label encoder
	training_data_y = le.transform(training_data_y)  # encode with label encoder

	# one-hot encoding
	training_data_y = np_utils.to_categorical(training_data_y)
	print training_data_y.shape
	return training_data_y


def load_test():
	path = "./data/test_data.npy"
	test_data_x = np.load(path)
	test_data_x = test_data_x.reshape(10000, 64*64)

	print("Test data x: {}".format(test_data_x.shape))

	return test_data_x


def main():
	training_x = load_x()
	training_y = load_y()
	dataset = Dataset(training_x, training_y)
	training, validation = dataset.split()
	nn = MultilayerNN([4096, 50, 40])
	nn.backprop(training, validation)
	test_x = load_test()
	prediction = nn.predict(test_x)
	with open("logistic_regression", 'w') as f:
		writer = csv.writer(f)
		writer.writerow(('Id', 'Label'))
		for i in range(1, len(prediction) + 1):
			writer.writerow((i, prediction[i - 1]))

main()