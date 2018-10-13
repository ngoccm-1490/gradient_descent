import numpy as np 
from numpy import linalg
np.random.seed(2)
X= np.random.rand(1,1000)
y =  3 + 10*X + np.random.randn(1,1000)
y = y.T
one = np.ones((1,1000))
Xbar = np.concatenate((X,one),axis =0 )
Xbar_T = Xbar.T
def grad(w = np.array((2,1))):
	return (1/X.shape[1])*np.dot(Xbar,(np.dot(Xbar_T,w)-y))
def cost_norm(w):
	return np.linalg.norm(w)/2
def optimizer_loss(w_init,rate,loop):
	x = []
	w = w_init
	for i in range(loop):
		w = w - rate* grad(w)
		x.append(w)
		if(len(x) > 2):
			if(cost_norm(np.dot(Xbar_T,x[-1])- np.dot(Xbar_T,x[-2]))/2< 0.0000000000000001):
				break
	return x[-1],i	
a,b = optimizer_loss(np.array([[8],[4]]),0.1,1000000)
print(a,b)		








