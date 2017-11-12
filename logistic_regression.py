from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import preprocessing
import csv


def load_x():
	path = "../data/training_data.npy"
	training_data_x = np.load(path)
	training_data_x = training_data_x.reshape(50000, 64*64)

	print("Training data x: {}".format(training_data_x.shape))

	return training_data_x


def load_y():
    path = "../data/train_y.csv"
    training_data_y = np.loadtxt(path, delimiter="\n")
    print training_data_y.shape
    return training_data_y


def load_test():
	path = "../data/test_data.npy"
	test_data_x = np.load(path)
	test_data_x = test_data_x.reshape(10000, 64*64)

	print("Test data x: {}".format(test_data_x.shape))

	return test_data_x



def main():
    clf = LogisticRegression()
    clf.fit(load_x(), load_y())
    prediction = clf.predict(load_test())
    with open("logistic_regression", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('Id', 'Label'))
        for i in range(1, len(prediction) + 1):
            writer.writerow((i, prediction[i - 1]))