import numpy as np
np.random.seed(1337)
testing = np.array([[1,2,3],[4,5,6],[7,8,9]])
thres = np.array([0.5,0.2,0.25])
testing = testing[:,thres < 0.3]

print(testing)