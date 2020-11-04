import numpy as np
import tensorflow as tf


a = np.array([[[1, 2, 3], [4, 5, 6], [4, 5, 6]]])
print(a)
print(a.shape)
a = np.reshape(a, (a.shape[0], -1))
print(a)
print(a.shape)

minst = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = minst.load_data()
num_train = 1500
mask = range(num_train)
X_train = X_train[mask]
y_train = y_train[mask]
print(X_train.shape)
ck, ck_num = np.unique(y_train, return_counts=True)
print(ck)
print(ck_num)

