import numpy as np
a = [-1, 1, 2, -4]
a = np.maximum(a, 0)
print(a)
print(np.sum(a))
b = [[1, 2, 3], [4, 5, 6]]
print(np.sum(b))

print(np.sum(b, axis=1))
