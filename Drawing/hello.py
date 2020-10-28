from sklearn.datasets import load_iris
import numpy as np
X_train, y_train = load_iris(return_X_y=True)
# print(X_train.shape)      # (150, 4)
# num_train = 150
num_dev = 5
# mask = np.random.choice(num_train, num_dev, replace=False)
mask = [40, 75, 25, 44, 76]
X = X_train[mask]
y = y_train[mask]
print(X.shape)      # (5, 4)
print(y.shape)      # (5,)
"""
[[7.  3.2 4.7 1.4]
 [6.4 3.2 5.3 2.3]
 [6.6 3.  4.4 1.4]
 [6.7 3.3 5.7 2.5]
 [6.8 2.8 4.8 1.4]]
"""
print(y)        # [1 2 1 2 1]

W = 0.001 * np.random.randn(4, 2)
# print(W)
scores = X.dot(W)
print(scores)
num_train = X.shape[0]
print(num_train)
print(np.arange(num_train))
correct_class_score = scores[np.arange(num_train),y]
print(correct_class_score)
print(correct_class_score.shape)
correct_class_score = np.reshape(correct_class_score,(num_train,-1))
print(correct_class_score)
print(correct_class_score.shape)
margins = scores - correct_class_score + 1
print(margins)
margins = np.maximum(0, margins)
print(margins)
margins[np.arange(num_train),y] = 0
print(margins)
margins[margins > 0] = 1
print(margins)
row_sum = np.sum(margins,axis=1)
print(row_sum)
margins[np.arange(num_train),y] = -row_sum.T
print(margins)
dW = np.dot(X.T,margins)
print(dW)