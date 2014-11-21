import numpy as np
import matplotlib.pyplot as plt 

data = np.loadtxt(open("girls_train.csv","rb"),delimiter=",")
transposed = np.transpose(data)

axes = [min(transposed[0]), max(transposed[0]), min(transposed[1]), max(transposed[1])]

plt.plot(transposed[0], transposed[1], 'ro')
plt.axis(axes)
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()