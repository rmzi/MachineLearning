import numpy as np
import matplotlib.pyplot as plt 

data = np.loadtxt(open("girls_train.csv","rb"),delimiter=",")

axes = [min(data.T[0]), max(data.T[0]), min(data.T[1]), max(data.T[1])]

plt.plot(data.T[0], data.T[1], 'ro')
plt.axis(axes)
plt.xlabel('Age')
plt.ylabel('Height')

## Gradient Descent Algorithm ##

# Mean-square Error
def computeError(betas, data):
	error = 0
	for i in range(0, data.shape[0]):
		error += (data[i][1] - (betas[1] * data[i][0] + betas[0])) ** 2
	return error / float(data.shape[0])

# Step function to update betas
def step(betas, data, learningRate, X):
	new_betas = [0,0]
	n = float(data.shape[0])
	b0_grad = 0 
	b1_grad = 0
	for i in range(0, int(n)):
		b0_grad += 1/n * (betas[0] + betas[1] * X[i][1] - data.T[1][i]) * X[i][0]
		b1_grad += 1/n * (betas[0] + betas[1] * X[i][1] - data.T[1][i]) * X[i][1]
	new_betas[0] = betas[0] - learningRate * b0_grad
	new_betas[1] = betas[1] - learningRate * b1_grad
	return new_betas

# Put it all together
def runGD(data, learningRate, iterations):
	# Setup Parameters
	y = data.T[1]
	X = np.column_stack((np.ones(len(y)), data.T[0]))
	betas = [0,0]

	for i in range(0, iterations):
		betas = step(betas, data, learningRate, X)

	return betas

results = runGD(data, 0.05, 1500)

print results
result_x = np.arange(0, max(data.T[0]), .1)
result_y = results[0] + results[1] * result_x

plt.plot(result_x, result_y)

print computeError(results, data)
plt.show()


