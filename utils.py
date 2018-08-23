from keras.datasets import mnist,cifar10
from keras.utils import to_categorical
import numpy as np
from skimage.util import random_noise


def load_mnist_drift(train_num=7000,test_num=1000,split=False):
	'''
	Return only training data.
	Include:
		- Mnist data
		- Gaussian Mnist data
		- Poisson Mnist data
		- Salt Mnist data

	Para:
		- train_num: number of training samples
		- test_num: number of test samples
		- split: return whole dataset or part of it
	'''

	num_classes = 10	
	(X_train,y_train),(X_test,y_test) =mnist.load_data()
	img_x, img_y = 28, 28
	input_shape = (img_x,img_y,1)
	X_train = X_train.reshape(X_train.shape[0],img_x,img_y,1)
	X_test = X_test.reshape(X_test.shape[0],img_x,img_y,1)
	y_train = to_categorical(y_train, num_classes)
	y_test = to_categorical(y_test,num_classes)

	if split:
		X_train,y_train = get_new_samples(X_train,y_train,train_num)
		X_test,y_test = get_new_samples(X_test,y_test,test_num)

	X_train_gaussian = random_noise(X_train,mode='gaussian',var=0.1)
	X_train_salt = random_noise(X_train,mode='salt',amount=0.1)
	X_test_gaussian = random_noise(X_test,mode='gaussian',var=0.1)
	X_test_salt = random_noise(X_test,mode='salt',amount=0.1)

	'''
	import matplotlib.pyplot as plt
	
	for img in [X_train,X_train_gaussian,X_train_poisson,X_train_salt]:
		
		fig = plt.figure()
		plt.imshow(img[0].reshape(img_x,img_y))
		
	plt.show()
	'''

	#I did not do preprocess because it will 
	#X_train = preprocess(X_train)
	#X_train_gaussian = preprocess(X_train_gaussian)
	#X_train_poisson = preprocess(X_train_poisson)
	#X_train_salt = preprocess(X_train_salt)

	X_train,y_train = np.concatenate((X_train,X_train_gaussian,X_train_salt)),np.concatenate((y_train,y_train,y_train))
	X_test,y_test = np.concatenate((X_test,X_test_gaussian,X_test_salt)),np.concatenate((y_test,y_test,y_test))

	return X_train,y_train,X_test,y_test

def preprocess(X):
	X = X.astype('float32')
	X /= 255
	return X


def get_new_samples(X,y,num):
	'Function to split the data'
	X_new = X[:num]
	y_new = y[:num]
	return X_new,y_new


