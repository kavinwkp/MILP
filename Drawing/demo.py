import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

a = np.array([[[1, 2, 3], [4, 5, 6], [4, 5, 6]]])
print(a)
print(a.shape)
a = np.reshape(a, (a.shape[0], -1))
print(a)
print(a.shape)

minst = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = minst.load_data()
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# 绘制数字：每张图像8*8像素点
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(X_train[i].reshape(28, 28), cmap='Greys', interpolation='nearest')
    # 用目标值标记图像
    ax.text(0, 7, str(y_train[i]))
plt.show()

