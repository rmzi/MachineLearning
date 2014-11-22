import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

data = np.loadtxt(open("girls_train.csv","rb"),delimiter=",")
test_data = np.loadtxt(open("girls_test.csv", "rb"),delimiter=",")

axes = [min(data.T[0]), max(data.T[0]), min(data.T[1]), max(data.T[1])]

# Part 1.1
# Plot data
plt.plot(data.T[0], data.T[1], 'ro')
plt.axis(axes)
plt.xlabel('Age')
plt.ylabel('Height')

# Part 1.2
## Gradient Descent Algorithm ##

# Mean-square Error (single feature)
def computeErrorSF(betas, data):
    error = 0
    for i in range(0, data.shape[0]):
        error += (data[i][1] - (betas[1] * data[i][0] + betas[0])) ** 2
    return error / float(data.shape[0])

# Step function to update betas (single feature)
def stepSF(betas, data, learningRate, X, y):
    new_betas = [0,0]
    n = float(data.shape[0])
    b0_grad = 0 
    b1_grad = 0
    for i in range(0, int(n)):
        # Calculate Gradients
        b0_grad += 1/n * (betas[0] + betas[1] * X[i][1] - y[i]) * X[i][0]
        b1_grad += 1/n * (betas[0] + betas[1] * X[i][1] - y[i]) * X[i][1]
    new_betas[0] = betas[0] - learningRate * b0_grad
    new_betas[1] = betas[1] - learningRate * b1_grad
    return new_betas

# Run Gradient Decent (single feature)
def runSFGD(data, learningRate, iterations):
    # Setup Parameters
    y = data.T[1]
    X = np.column_stack((np.ones(len(y)), data.T[0]))
    betas = [0,0]

    # Iterate algorithm (learningRate) times
    for i in range(0, iterations):
        betas = stepSF(betas, data, learningRate, X, y)

    return betas

results = runSFGD(data, 0.05, 1500)

# Resulting Betas
print "[beta0, beta1]"
print results

# Compute error for training data
print "Error of training data"
print computeErrorSF(results, data)

# Part 1.3
# Setup Plotting Regression Line
result_x = np.arange(0, max(data.T[0]), .1)
result_y = results[0] + results[1] * result_x

plt.plot(result_x, result_y)
#plt.show()

# Part 1.4
# Testing model on new data

# Compute error for test data
print "Error of test data"
print computeErrorSF(results, test_data)

# Part 2.1
# Prep data

new_data = np.loadtxt(open("girls_age_weight_height_2_8.csv","rb"),delimiter=",")

# Standard Deviation of Each Feature
print "Standard Deviation / Mean of Features"

std_age = np.std(new_data.T[0])
mean_age = np.mean(new_data.T[0])
print "Age: ", std_age, " ", mean_age

std_weight = np.std(new_data.T[1])
mean_weight = np.mean(new_data.T[1])
print "Weight: ", std_weight, " ", mean_weight

std_height = np.std(new_data.T[2])
mean_height = np.mean(new_data.T[2])
print "Height: ", std_height, " ", mean_height

# Scale each feature
def scaleVector(vector, mean, std):
    for i in range(0,len(vector)):
        vector[i] = (vector[i] - mean) / std
    return vector

scaled_age = scaleVector(new_data.T[0], mean_age, std_age)
#print scaled_age

scaled_weight = scaleVector(new_data.T[1], mean_weight, std_weight)
#print scaled_weight

scaled_height = scaleVector(new_data.T[2], mean_height, std_height)
#print scaled_height

# Step function to update betas (multi-feature)
def stepMF(betas, data, learningRate, X, y):
    new_betas = [0,0, 0]
    n = float(data.shape[0])
    b0_grad = 0 
    b1_grad = 0
    b2_grad = 0
    for i in range(0, int(n)):
        # Calculate Gradients
        b0_grad += 1/n * (betas[0] + betas[1] * X[i][1] + betas[2] * X[i][2] - y[i]) * X[i][0]
        b1_grad += 1/n * (betas[0] + betas[1] * X[i][1] + betas[2] * X[i][2] - y[i]) * X[i][1]
        b2_grad += 1/n * (betas[0] + betas[1] * X[i][1] + betas[2] * X[i][2] - y[i]) * X[i][2] 
    new_betas[0] = betas[0] - learningRate * b0_grad
    new_betas[1] = betas[1] - learningRate * b1_grad
    new_betas[2] = betas[2] - learningRate * b2_grad
    return new_betas

# Run Gradient Decent (multi-feature)
def runMFGD(data, learningRate, iterations):
    # Setup parameters
    y = data.T[2]
    X = np.column_stack((np.ones(len(y)), data.T[0], data.T[1]))
    betas = [0,0,0]

    # Iterate algorithm (learningRate) times
    for i in range(0, iterations):
        betas = stepMF(betas, data, learningRate, X, y)

    return betas

alpha1 = runMFGD(new_data, 0.005, 50)
alpha2 = runMFGD(new_data, 0.001, 50)
alpha3 = runMFGD(new_data, 0.05, 50)
alpha4 = runMFGD(new_data, 0.1, 50)
alpha5 = runMFGD(new_data, 0.5, 50)
alpha6 = runMFGD(new_data, 1, 50)

print alpha1, "Learning Rate: 0.005, Iterations: 50"
print alpha2, "Learning Rate: 0.001, Iterations: 50"
print alpha3, "Learning Rate: 0.05, Iterations: 50"
print alpha4, "Learning Rate: 0.1, Iterations: 50"
print alpha5, "Learning Rate: 0.5, Iterations: 50"
print alpha6, "Learning Rate: 1, Iterations: 50"

# Plot Results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(scaled_age, scaled_weight, scaled_height)

ax.set_xlabel('Age')
ax.set_ylabel('Weight')
ax.set_zlabel('Height')

plt.show()


