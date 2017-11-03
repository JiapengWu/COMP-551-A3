Please put the training data and test data into a folder called "data"

Here are the list of rules that might be useful during the data processing. If you can think of any, put them down here:
1. if there are 2 3s and there are no 'A' in the output, treat one of the 3 as 'M'

To run the preprocessing:
- Create folder "data" and move your training set, training label and training test there.
- Go to the file `preprocessing.py`. If it's the first time your run it, be sure to call the function "partition_trainin_set" with default parameters(no parameters passed).
- Check out the function list, there are bunch of function either for loading data, showing images or playing around with.
- Run "filter" on the training images for binarization.