'''
batch_size:		number of training examples / iteration
num_epochs:		number of training epochs
kernel_size:	Kernel size
pool_size:		Pooling Size
conv_depth_1:	Number of kernels per convolutional layer (1st type)
conv_depth_2: 	Number of kernels per convolutional layer (2nd type)
drop_prob_1:	Dropout prob after pooling
drop_prob_2:	Dropout prob in Fully Connected (dense) layer
hidden_size:	# neurons in dense layers
'''

batch_size = 32
num_epochs = 300
kernel_size = 3 # use 3x3 kernels
pool_size = 2 # 2x2 pooling
conv_depth_1 = 32 #start with 32 kernels
conv_depth_2 = 64 #then 64 kernels per layer (after pooling)
drop_prob_1 = 0.25 
drop_prob_2 = 0.5 
hidden_size = 512 # the FC layer will have 512 neurons