import numpy as np 
def cost(x,y):
	return (x-1)**2 + (y-2)**2 + 3
def grad_for_x(x,y):
	return 2*(x-1)
def grad_for_y(x,y):
	return 2*(y-2)
def update_gradient(x_init,y_init,learning_rate):
	x = x_init
	y = y_init
	X = []
	Y = []
	for i in range(1000000):
		x = x - learning_rate * grad_for_x(x,y)
		y = y - learning_rate * grad_for_y(x,y)
		X.append(x)
		Y.append(y)
		if(i >2):
			if(cost(X[-1],Y[-1])- cost(X[-2],Y[-2])> -0.000001 and cost(X[-1],Y[-1])- cost(X[-2],Y[-2])<0.000001):
				break
	return X[-1],Y[-1],i

x,y,i = update_gradient(3,3,0.001)
print(x,y,i)				

