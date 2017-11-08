import numpy   as np
import scipy.misc  # to visualize only
from matplotlib import pyplot as plt
import linecache
import parameter as p
from copy import deepcopy
from scipy import ndimage


# load a single image from either training or test dataset, index specified by n
def load_single_image(dataset = "train", n = 10):
    l = linecache.getline('../data/' + dataset + '_x.csv', n)
    x = [float(n.strip()) for n in l.split(',')]
    x = np.array(x).reshape(64, 64)
    return x


# partition 50000 pictures into 50000/n subsets
def partition_trainin_set(n = 100, size = 50000):
    with open('../data/train_x.csv', "r") as f:
        partitions = size/n
        for k in range(partitions):
            out = open('../data/sample' + str(k+1) + '.csv', 'w')
            for i in range(n):
                image = f.readline()
                out.write(image)


# load all images from a sample csv indicated by n
def load_sample_images(n = 2):
    x = np.loadtxt('../data/sample' + str(n) + '.csv', delimiter=",")  # load from text
    x = x.reshape(-1, 64, 64)  # reshape
    return x


# show number of "amount" images from a sample csv indicated by n
def show_sample(n, amount = 100):
    x = np.loadtxt('../data/sample' + str(n) + '.csv', delimiter=",")  # load from text
    x = x.reshape(-1, 64, 64)  # reshape
    for i in range(amount):
        plt.imshow(x[i])  # to visualize only
        plt.show()


def show_images(x):
    if len(x.shape) == 2:
        plt.imshow(x)  # to visualize only
        plt.show()
    else:
        for i in range(x.shape[0]):
            plt.imshow(x[i])  # to visualize only
            plt.show()


def binarization(x, thresh = 225):
    y = np.copy(x)
    high_values = y >= thresh
    low_values = y < thresh  # Where values are low
    y[low_values] = 0
    y[high_values] = 1
    return y


def filter(x, func, param):
    y = np.copy(x)
    for i in range(y.shape[0]):
        y[i] = func(x[i], param)
    return y


# show two given sets of images
def compare_results(x1, x2, single=False):
    if single:
        plt.subplot(121)
        plt.imshow(x1)  # to visualize only
        plt.subplot(122)
        plt.imshow(x2)
        plt.show()
    else:
        for i in range(x1.shape[0]):
            plt.subplot(121)
            plt.imshow(x1[i])  # to visualize only
            plt.subplot(122)
            plt.imshow(x2[i])
            plt.show()


# this function shall be only called when adopting a different blur function or parameter
# than the one used last time
def pipeline():
    path = "../data/preprocessed_samples/Training_X_"
    result = []
    for i in range(500):
        x = load_sample_images(i + 1)
        blurred_x = filter(x, p.blur_function, p.blur_parameter)
        y = binarization(blurred_x, p.thresh_hold)
        result.append(np.copy(y))
    training_data = np.array(result)
    np.save("../data/training_data", training_data)
    return training_data


def preprocess_test():
    x = np.loadtxt('../data/test_x.csv', delimiter=",")  # load from text
    x = x.reshape(-1, 64, 64)  # reshape
    test_data = np.array([binarization(filter(x[i], p.blur_function, p.blur_parameter), p.thresh_hold) for i in range(x.shape[0])])
    np.save("../data/test_data", test_data)
    return test_data


if __name__ == '__main__':
    # partition_trainin_set(100)
    # show_test(2)
    x = load_sample_images(8)
    blurred_x1 = filter(x, ndimage.median_filter, 2)

    blurred_x2 = filter(x, ndimage.maximum_filter, 2)
    # compare_results(x, blurred_x)

    y1 = binarization(blurred_x2, p.thresh_hold)
    y2 = binarization(blurred_x1, p.thresh_hold)
    compare_results(y1, y2)

    # open_square = ndimage.binary_opening(y)
    # eroded_square = ndimage.binary_erosion(y)
    # reconstruction = ndimage.binary_propagation(eroded_square, mask=y)
    # compare_results(y, reconstruction)

    # training_data = pipeline()
    # test_data = preprocess_test()
    # print test_data.shape


