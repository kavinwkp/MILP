import numpy as np
import tensorflow as tf
from neural_net import TwoLayerNet
import time
from PIL import Image
import matplotlib.pyplot as plt


class NeuralNetwork():

    def __init__(self):
        self.input_size = 28 * 28
        self.hidden_size = 50
        self.num_classes = 10
        self.net = TwoLayerNet(self.input_size, self.hidden_size, self.num_classes)
        self.fit()

    def fit(self):
        minst = tf.keras.datasets.mnist
        (X_train, y_train), (X_test, y_test) = minst.load_data()
        num_train = 30000
        num_val = 1000

        # Train set
        mask = list(range(num_train, num_train + num_val))
        X_val = X_train[mask]
        y_val = y_train[mask]

        mask = list(range(num_train))
        X_train = X_train[mask]
        y_train = y_train[mask]

        # Preprocessing: reshape the images data into rows
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_val = np.reshape(X_val, (X_val.shape[0], -1))
        stats = self.net.train(X_train, y_train, X_val, y_val,
                               num_iters=1500, batch_size=200,
                               learning_rate=1e-4, learning_rate_decay=0.95,
                               reg=0.25, verbose=True)
        # val_acc = (self.net.predict(X_val) == y_val).mean()
        # print('Validation accuracy: ', val_acc)
        # plt.subplot(2, 1, 1)
        # plt.plot(stats['loss_history'])
        # plt.title('Loss history')
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(stats['train_acc_history'], label='train')
        # plt.plot(stats['val_acc_history'], label='val')
        # plt.title('Classification accuracy history')
        # plt.xlabel('Epoch')
        # plt.ylabel('Clasification accuracy')
        # plt.show()

    def predict(self, x):
        return self.net.predict(x)


def main():
    model = NeuralNetwork()
    # img = Image.open("./images/test6.png")
    # img_array = np.array(img.convert('L')).reshape(1, 784)
    # result = model.predict(img_array)
    # print(result)

    minst = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = minst.load_data()
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    result = model.predict(X_test)
    print("result: %f" % np.mean(result == y_test), )
    num = [0] * 10
    count = [0] * 10
    num_test = X_test.shape[0]
    for i in range(num_test):
        num[y_test[i]] += 1
        if result[i] == y_test[i]:
            count[y_test[i]] += 1
    print(count)
    print(num)

    print("result: %f" % (np.sum(count) / np.sum(num)))     # 0.865100

    accurate = list()
    for i in range(10):
        accurate.append(count[i] / num[i])
    print(np.around(accurate, 3))
    # [0.972 0.979 0.846 0.848 0.874 0.733 0.91  0.866 0.79  0.808]

if __name__ == '__main__':
    main()






