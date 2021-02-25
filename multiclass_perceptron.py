import numpy as np
import math
import perceptron

from sklearn.datasets import load_svmlight_file
#data=load_svmlight_file(sys.argv[1])

"""
x is the test data

w is the weight vector

bias is the offset or threshold. No need to set this if you used the
learning() function

returns: 1 for class 1 and -1 for class 2
"""
def activation(x, w, bias=0):
	x=_prepend_test_feature(x)
	print("\n\n")
	print(str(x))
	print(str(w))
	if(np.dot(x,w)-bias>0):
		return 1
	return -1
	

"""
x is the training set which is an NxD matrix

y is the 1xD set of labels corresponding to the training set
y is either 1 or -1

limit is factor of 10000 for multiples of training iterations.
limit=1 means there will be 10000 iterations, 2 will be 2000 and so on
This is used because the training process doesn't converge for non-linearly
separable data

step size is a constant used for updating the weights

returns: weight vector to use as input on activation function
"""
def learning(fname, limit,num_classes, step_size, bias=0):
	data=load_svmlight_file(fname)
	samples=data[0].get_shape()[0]
	features=data[0].get_shape()[1]
	
		
	for j in range(num_classes-1):#need to do more with this
		limit_count=0
		while limit_count < limit*10000:
			y_hat=0
			count=0	#counts the amount of updates 
			for i in range(x.shape[0]):
				y_hat=np.dot(w,x[i])
				if y_hat-bias > 0:
					y_hat=1
				else:
					y_hat=-1
				if not y[i] == y_hat:
					w=w+step_size*y[i]*x[i]
					count+=1
			if count==0:#check if w has converged
				print("converged!")
				break
			limit_count+=1
		if limit_count == limit*1000:
			print("did not converge")
		return w


"""
this is used to prepend a column of 1s to the training set
in order to to avoid using a separate bias. 

only use this on the training data

x is the input data in the form of a numpy array. 
Can be either 1D or 2D. if 1D this function will convert it
"""
def _prepend_trng_feature(x):
	a=np.ones((x.shape[0],1))
	print(str(x.ndim)+ " dimensions")
	if x.ndim > 1:
		return np.hstack((a,x))
	else:
		return np.hstack((a,np.array(x)[np.newaxis].T))

def _prepend_test_feature(x):
	return np.insert(x, 0, 1.,axis=0)



