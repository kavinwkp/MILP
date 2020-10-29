import numpy as np
import tensorflow as tf
from linear_classifier import LinearSVM
import time
from PIL import Image

class DigitClassifier():

    def __init__(self):
        self.svm = LinearSVM()
        self.fit()

    def fit(self):
        minst = tf.keras.datasets.mnist
        (X_train, y_train), (X_test, y_test) = minst.load_data()
        num_train = 15000
        num_val = 1000
        num_dev = 500
        num_test = 10000

        # Validation set
        mask = range(num_train, num_train + num_val)
        X_val = X_train[mask]
        y_val = y_train[mask]

        # Train set
        mask = range(num_train)
        X_train = X_train[mask]
        y_train = y_train[mask]

        # Small training set (development set)
        mask = np.random.choice(num_train, num_dev, replace=False)
        X_dev = X_train[mask]
        y_dev = y_train[mask]

        # Preprocessing: reshape the images data into rows
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_val = np.reshape(X_val, (X_val.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
        X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
        X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
        tic = time.time()
        loss_hist = self.svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                              num_iters=1500, verbose=True)
        toc = time.time()
        print('That took %fs' % (toc - tic))

    def predict(self, x):
        return self.svm.predict(x)


def main():
    model = DigitClassifier()
    # img = Image.open("./images/test6.png")
    # img_array = np.array(img.convert('L')).reshape(784)
    # img_array = np.hstack([img_array, [1.0]]).reshape((1, 785))
    # result = model.predict(img_array)
    minst = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = minst.load_data()

    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    result = model.predict(X_test)
    print("result: %f" % np.mean(result == y_test), )
    num = list()
    for i in range(10):
        num.append(0)
    count = list()
    for i in range(10):
        count.append(0)
    num_test = X_test.shape[0]
    for i in range(num_test):
        num[y_test[i]] += 1
        if result[i] == y_test[i]:
            count[y_test[i]] += 1
    print(count)
    print(num)

    print("result: %f" % (np.sum(count) / np.sum(num)))

    accurate = list()
    for i in range(10):
        accurate.append(count[i] / num[i])      # 0.889500
    print(np.around(accurate, 3))
    # [0.974 0.974 0.819 0.864 0.882 0.832 0.922 0.893 0.841 0.88]

if __name__ == '__main__':
    main()






