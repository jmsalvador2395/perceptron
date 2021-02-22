import numpy as np
import math



"""
x is the test data

w is the weight vector

bias is the offset or threshold. don't set this if 

returns: 1 for class 1 and -1 for class 2
"""
def activation(x, w, bias=0):
	if(np.dot(x,w)-bias>0):
		return 1
	return -1
	

"""
x is the training set which is an NxD matrix

y is the 1xD set of labels corresponding to the training set
y is either 1 or -1

limit is factor of 1000 for multiples of training iterations.
limit=1 means there will be 1000 iterations, 2 will be 2000 and so on
This is used because the training process doesn't converge for non-linearly
separable data

step size is a constant used for updating the weights

returns: weight vector to use as input on activation function
"""
def learning(x, y, limit, step_size, bias=0):
	if x.ndim > 1:
		w=np.zeros(x.shape[1])
	else:
		w=np.array([1])
		
	j=0
	while j < limit*1000:
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
		j+=1
	if j == limit*1000:
		print("did not converge")
	return w


"""
this is used to prepend a column of 1s to avoid using a
separate bias. 

x is the input data in the form of a numpy array. 
Can be either 1D or 2D. if 1D this function will convert it
"""
def prepend_feature(x):
	a=np.ones((x.shape[0],1))
	if x.ndim > 1:
		return np.hstack((a,x))
	else:
		return np.hstack((a,np.array(x)[np.newaxis].T))