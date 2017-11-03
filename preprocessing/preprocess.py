import numpy   as np
import scipy.misc  # to visualize only
from matplotlib import pyplot as plt
import linecache
import parameter as p

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


def filter_image(x, thresh = 225):
    y = np.copy(x)
    high_values = y >= thresh
    low_values = y < thresh  # Where values are low
    y[low_values] = -0.1
    y[high_values] = 1.175
    return y


# show two given sets of images
def compare_results(x1, x2):
    for i in range(x1.shape[0]):
        plt.subplot(121)
        plt.imshow(x1[i])  # to visualize only
        plt.subplot(122)
        plt.imshow(x2[i])
        plt.show()


if __name__ == '__main__':
    # partition_trainin_set(100)
    # show_test(2)
    x = load_sample_images(8)
    # show_images(x)
    y = filter_image(x, p.thresh_hold)
    compare_results(x, y)

