import numpy as np
a = np.array([[[1, 2, 3], [4, 5, 6], [4, 5, 6]]])
print(a)
print(a.shape)
a = np.reshape(a, (a.shape[0], -1))
print(a)
print(a.shape)