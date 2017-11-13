import numpy   as np
#import scipy.misc  # to visualize only
# from matplotlib import pyplot as plt
import linecache
import parameter as p
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
def pipeline(filename='', blur_function=None, blur_parameter=None):
    training_data = np.ones((100, 64, 64))
    # default : do nothing, stores data only
    if blur_function is None:
        for i in range(500):
            x = load_sample_images(i + 1)
            if i == 0:
                training_data = x
            else:
                training_data = np.concatenate((training_data, x))
        np.save("../data/training_data", training_data)
    else:
        for i in range(500):
            x = load_sample_images(i + 1)
            blurred_x = filter(x, blur_function, blur_parameter)
            y = binarization(blurred_x, p.thresh_hold)
            if i == 0:
                training_data = y
            else:
                training_data = np.concatenate((training_data, y))
        np.save("../data/training_data_" + filename, training_data)
    return training_data


def preprocess_test(filename='', blur_function=None, blur_parameter=None):
    x = np.loadtxt('../data/test_x.csv', delimiter=",")  # load from text
    x = x.reshape(-1, 64, 64)  # reshape
    if blur_function is None:
        np.save("../data/test_data", x)
    else:
        test_data = np.array(
            [binarization(filter(x[i], blur_function, blur_parameter), p.thresh_hold) for i in range(x.shape[0])])

        np.save("../data/test_data_" + filename, test_data)
    return test_data


if __name__ == '__main__':
    parameter_grid = zip(p.filename, p.blur_function)
    map(lambda x: pipeline(x[0], x[1], p.blur_parameter), parameter_grid)
    map(lambda x: preprocess_test(x[0], x[1], p.blur_parameter), parameter_grid)
    pipeline()
    preprocess_test()

