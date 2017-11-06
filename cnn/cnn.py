from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from CNNParameters import batch_size, num_epochs, kernel_size, pool_size, conv_depth_1, conv_depth_2, drop_prob_1, drop_prob_2, hidden_size
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint,EarlyStopping
import csv
import numpy as np

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
output_path = "test_cnn.h5"

#prediction output path : Final prediction for Kaggle submission
prediction_filepath = "prediction.csv"

#where we will save the weights
weights_filepath="weights.best.hdf5"	


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
	training_data_y = np.loadtxt(y_train_file, delimiter="\n")
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

	print("Training data y: {}".format(training_data.shape))

	return training_data_y

def load_test():
	path = "../data/test_data.npy"
	test_data_x = np.load(path)
	test_data_x = test_data_x.reshape(50000, 64, 64, 1)

	print("Test data x: {}".format(test_data_x.shape))

	return test_data_x

#DEPRECATED
#Returns (x_train, x_test, y_train)
def load_inputs():

	#loading and reshaping Training data
	#x_train = np.loadtxt(x_train_file, delimiter=",") # load from text 
	#x_train = x_train.reshape(x_train.shape[0], 64, 64, 1) # reshape 

	#loading, preprocess training data
	y_train = np.loadtxt(y_train_file, delimiter="\n")
	#y_train = np_utils.to_categorical(y_train)

	#Label Encode our results.
	#[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.
	#15.  16.  17.  18.  20.  21.  24.  25.  27.  28.  30.  32.  35.  36.  40.
	#42.  45.  48.  49.  54.  56.  63.  64.  72.  81.]
	#will be encoded to 0,1,2,3,...
	le = preprocessing.LabelEncoder()


	le.fit(y_train)		#fit with label encoder
	y_train = le.transform(y_train)		#encode with label encoder

	#one-hot encoding
	y_train = np_utils.to_categorical(y_train)

	print("Number of classes: {}".format(len(le.classes_)))

	#loading and reshaping Testing data
	#x_test = np.loadtxt(x_test_file, delimiter=",") # load from text 
	#x_test = x_test.reshape(x_test.shape[0], 64, 64, 1) # reshape 

	#normalize data
	#X_train = x_train.astype('float32') 
	#X_test = x_test.astype('float32')
	#X_train /= np.max(X_train) # Normalise data to [0, 1] range
	#X_test /= np.max(X_test) # Normalise data to [0, 1] range

	#print(X_train[8], y_train[8])
	#print("Shape of Train -X: {}".format(x_train.shape))
	#print("Shape of Train -Y: {}".format(y_train.shape))
	#print("Shape of Testing : {}".format(x_test.shape))


	print("Data loaded Successfully")
	return(y_train)


#Builds the cnn, returns the model
def build_cnn(X_train, Y_train):

	print("Starting CNN")

	#input = input
	num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10 
	#num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
	num_classes = 40 # there are 10 image classes
	inp = Input(shape=(height, width, depth))
	 
	# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
	conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
	conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
	pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
	drop_1 = Dropout(drop_prob_1)(pool_1)

	# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
	conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
	conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
	pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
	drop_2 = Dropout(drop_prob_1)(pool_2)

	# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
	flat = Flatten()(drop_2)
	hidden = Dense(num_classes, activation='relu')(flat)
	drop_3 = Dropout(drop_prob_2)(hidden)
	out = Dense(num_classes, activation='softmax')(drop_3)

	model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

	model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
				  optimizer='adam', # using the Adam optimiser
				  metrics=['accuracy']) # reporting the accuracy
	return model


#takes inputs and model, and trains model. Saves weights and model
def train_cnn(X_train, Y_train, model):

	#callbacks and saving
	checkpoint = ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	stopping = EarlyStopping(monitor='val_acc', min_delta=0.0007, patience=10, verbose=1, mode='auto')
	callbacks_list = [checkpoint]

	print("Training CNN")
	model.fit(X_train, Y_train,                # Train the model using the training set...
			  batch_size=batch_size, epochs=num_epochs,
			  verbose=1,callbacks = callbacks_list, validation_split=0.1) # ...holding out 10% of the data for validation

	#save the model
	model.save(output_path)
	print("Model trained and saved as {}".format(output_path))

#loads a cnn with specified weights, returns model
def load_cnn(weights_filepath):

	#build model like before
	return build_cnn(load_x(), load_y()).load_weights(weights_filepath)


#Predicts our test set and outputs results in csv format
def predict(model):

	#load our test set
	x_test = load_test()

	#use model to predict our test set
	pred = model.predict(x_test,verbose = 1)

	#results are one-hot encoded, we'll decode back
	predictions = le.inverse_transform(pred)

	#save to csv
	with open(prediction_filepath, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(('Id', 'Category'))
		for i in range(1,len(predictions)+1):
			writer.writerow((i, predictions[i-1]))

	print("Predictions Successfully saved in {}".format(prediction_filepath))
	 


if __name__ == '__main__':
	#Note: For the binary model, we can bypass
	train_x = load_binary_input_x()

	train_y = load_inputs() 
	train_cnn(train_x, train_y)
