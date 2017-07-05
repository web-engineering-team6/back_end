#coding:utf-8

import numpy as np
from scipy import ndimage

rng = np.random.RandomState(1234)

def cv2input(x):
	input_x = x.copy{().trancpose()###############
	return input_x


def flipping(x):
	x_return = x.copy()
	mask = (np.random.randint(0, 2, len(x)) == 1)
	x_return[mask] = x_return[mask][:, :, ::-1, :]
	return x_return

	
def cropping(x):
	padded = np.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='constant')
	crops = rng.randint(8, size=(len(x), 2))
	cropped_train_X = [padded[i, c[0]:(c[0]+32), c[1]:(c[1]+32), :] for i, c in enumerate(crops)]
	cropped_train_X = np.array(cropped_train_X)
	return cropped_train_X
	
def rolling(x):
	r = rng.randint(-40, 41)
	rol_x = ndimage.rotate(x, r, reshape=False)
	return rol_x
	

def gcn(x):
	mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
	std = np.std(x, axis=(1, 2, 3), keepdims=True)
	return (x - mean)/std
	
	
class ZCAWhitening:
	def __init__(self, epsilon=1e-4):
		self.epsilon = epsilon
    	self.mean = None
    	self.ZCA_matrix = None

	def fit(self, x):
    	x = x.reshape(x.shape[0], -1)
    	self.mean = np.mean(x, axis=0)
    	x -= self.mean
    	cov_matrix = np.dot(x.T, x) / x.shape[0]
    	A, d, _ = np.linalg.svd(cov_matrix)
    	self.ZCA_matrix = np.dot(np.dot(A, np.diag(1. / np.sqrt(d + self.epsilon))), A.T)

	def transform(self, x):
    	shape = x.shape
    	x = x.reshape(x.shape[0], -1)
    	x -= self.mean
    	x = np.dot(x, self.ZCA_matrix.T)
    	return x.reshape(shape)