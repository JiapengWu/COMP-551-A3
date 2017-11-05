from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from CNNParameters import batch_size, num_epochs, kernel_size, pool_size, conv_depth_1, conv_depth_2, drop_prob_1, drop_prob_2, hidden_size
from sklearn import preprocessing

import numpy as np

'''
	1. Run first_time_pipeline() in ../preprocessing/preprocess.py
	2. Run 
'''


x_train_file = open('../data/sample1.csv')
#x_train_file = open('../data/train_x.csv')
y_train_file = open('../data/train_y.csv')
x_test_file = open('../data/test_x.csv')


#model output path
output_path = "test_cnn.h5"

#prediction output path
prediction_file_path = "prediction.csv"


#This is the pipeline function from preprocessing
#returns (50000, 64, 64, 1) shape np array
def load_binary_input_x():
    path = "../data/preprocessed_samples/Training_X_"
    training_data = np.array([np.load(path + str(i + 1) + '.npy') for i in range(500)])
    training_data = training_data.reshape(50000, 64, 64, 1)
    return training_data


#loads files
#Returns (x_train, x_test, y_train)
def load_inputs():

	#loading and reshaping Training data
	x_train = np.loadtxt(x_train_file, delimiter=",") # load from text 
	x_train = x_train.reshape(x_train.shape[0], 64, 64, 1) # reshape 

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
	x_test = np.loadtxt(x_test_file, delimiter=",") # load from text 
	x_test = x_test.reshape(x_test.shape[0], 64, 64, 1) # reshape 

	#normalize data
	X_train = x_train.astype('float32') 
	X_test = x_test.astype('float32')
	X_train /= np.max(X_train) # Normalise data to [0, 1] range
	X_test /= np.max(X_test) # Normalise data to [0, 1] range

	#print(X_train[8], y_train[8])
	print("Shape of Train -X: {}".format(x_train.shape))
	print("Shape of Train -Y: {}".format(y_train.shape))
	print("Shape of Testing : {}".format(x_test.shape))


	print("Data loaded Successfully")
	return(X_train, y_train, X_test)


#Trains the CNN, outputs the model to a .h5 file
def train_cnn(X_train, Y_train):

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

	model.fit(X_train, Y_train,                # Train the model using the training set...
	          batch_size=batch_size, epochs=num_epochs,
	          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
	#model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

	#save the model
	model.save(output_path)
	print("Model trained and saved as {}".format(output_path))

#Predicts our test set and outputs results in csv format
def validate_cnn(x_test):
	#load our trained model
	model.load_model(output_path)

	#use model to predict our test set
	pred = model.predict(x_test,verbose = 1)

	#results are one-hot encoded, we'll decode back
	predictions = le.inverse_transform(pred)

	#save to csv
	np.savetxt(prediction_file_path, delimiter = '\n')

if __name__ == '__main__':
	#Note: For the binary model, we can bypass

	train_x, train_y, test_x = load_inputs() 
	cnn(train_x, train_y)
