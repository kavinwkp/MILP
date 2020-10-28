import numpy as np
a = [1.1, 3.3, 5.5]
y = [1, 2, 0]
scores = np.array([[1.1, 3.3, 5.5],
                   [1.1, 3.3, 5.5],
                   [1.1, 3.3, 5.5]])
num = scores.shape[0]

x = scores[np.arange(3),y]
print(x)