from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from CNNParameters import batch_size, num_epochs, kernel_size, pool_size, conv_depth_1, conv_depth_2, drop_prob_1, drop_prob_2, hidden_size
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint,EarlyStopping
import csv
import numpy as np
from resnet import ResnetBuilder

'''
	0. Make sure training and testing data are in ../data
	1. Run:
			load_x() to load training x
			load_y() to load training y
'''


#x_train_file = open('../data/sample1.csv')
#x_train_file = open('../data/train_x.csv')
#y_train_file = open('../data/train_y.csv')
#x_test_file = open('../data/test_x.csv')


#model output path : This stores the latest model
output_path = "resnet.h5"

#prediction output path : Final prediction for Kaggle submission
prediction_filepath = "resnet.prediction.csv"

#where we will save the weights
weights_filepath="resnet.weights.best.hdf5"	


#This is the pipeline function from preprocessing
#returns (50000, 64, 64, 1) shape np array
def load_x():
	path = "../data/training_data.npy"
	training_data_x = np.load(path)
	training_data_x = training_data_x.reshape(50000, 64, 64, 1)

	print("Training data x: {}".format(training_data_x.shape))

	return training_data_x

def load_y():
	path = "../data/train_y.csv"
	training_data_y = np.loadtxt(path, delimiter="\n")
	#Label Encode our results.
	#[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.
	#15.  16.  17.  18.  20.  21.  24.  25.  27.  28.  30.  32.  35.  36.  40.
	#42.  45.  48.  49.  54.  56.  63.  64.  72.  81.]
	#will be encoded to 0,1,2,3,...
	#this means that "81" will be [0,0,0,0,0,...,0,1] where the 40th entry is 1
	le = preprocessing.LabelEncoder()

	le.fit(training_data_y)		#fit with label encoder
	training_data_y = le.transform(training_data_y)		#encode with label encoder

	#one-hot encoding
	training_data_y = np_utils.to_categorical(training_data_y)

	print("Training data y: {}".format(training_data_y.shape))

	return training_data_y, le

def build_resnet():
	input_shape = (1, 64, 64)
	output_shape = 40

	model = ResnetBuilder.build_resnet_18(input_shape,output_shape)
	model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
				  optimizer='adam', # using the Adam optimiser
				  metrics=['accuracy']) # reporting the accuracy
	return model

def train_resnet():
	model = build_resnet()
	x_train = load_x()

	y_train,a = load_y()


	checkpoint = ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	stopping = EarlyStopping(monitor='cc', min_delta=0.001, patience=10, verbose=1, mode='auto')
	callbacks_list = [checkpoint]

	print("Training ResNet")
	model.fit(x_train, y_train,                # Train the model using the training set...
			  batch_size=32, epochs=10,
			  verbose=1,callbacks = callbacks_list, validation_split=0.1) # ...holding out 10% of the data for validation


if __name__ == '__main__':
	train_resnet()
