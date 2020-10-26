from collections import defaultdict
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer



class NaiveBayesScratch():
    """朴素贝叶斯算法Scratch实现"""
    def __init__(self):
        # 先验概率 P(Y=ck)
        self._prior_prob = defaultdict(float)
        # 似然概率 P(X|Y=ck)
        self._likelihood = defaultdict(defaultdict)
        # 每个类别的样本在训练集中出现次数
        self._ck_counter = defaultdict(float)
        # 每一个特征可能取值的个数
        self._Sj = defaultdict(float)

    def fit(self, X, y):

        n_sample, n_feature = X.shape

        # value and counter of each category
        ck, ck_num = np.unique(y, return_counts=True)
        # ck: [0 1 2] 有哪些类
        # ck_num: 每类有多少个

        # number of category
        self._ck_counter = dict(zip(ck, ck_num))
        # print(self._ck_counter)
        # calculate prior probability
        # (number of label + lambda) / (number of sample + number of category * lambda)
        for label, label_num in self._ck_counter.items():
            self._prior_prob[label] = (label_num + 1) / (n_sample + ck.shape[0])
        # print(prior)

        # index of each category list(list)
        ck_idx = list()

        for label in ck:
            # all index of one each category 每一类的index
            label_idx = np.squeeze(np.argwhere(y == label))
            ck_idx.append(label_idx)

        # each label
        for label, idx in zip(ck, ck_idx):
            # all sample of one label   取出每一类的所有样本
            xdata = X[idx]
            label_likelihood = defaultdict(defaultdict)
            for i in range(n_feature):
                # probability of each dimension of feature
                feature_val_prob = defaultdict(float)
                # value and number of each feature
                feature_val, feature_cnt = np.unique(xdata[:, i], return_counts=True)
                self._Sj[i] = feature_val.shape[0]
                for fea_val, cnt in zip(feature_val, feature_cnt):
                    feature_val_prob[fea_val] = (cnt + 1) / (self._ck_counter[label] + self._Sj[i])
                label_likelihood[i] = feature_val_prob
            self._likelihood[label] = label_likelihood

    def predict(self, x):
        # x is each test
        post_prob = defaultdict(float)
        for label, label_likelihood in self._likelihood.items():
            prob = np.log(self._prior_prob[label])

            for i, fea_val in enumerate(x):
                feature_val_prob = label_likelihood[i]
                if fea_val in feature_val_prob:
                    prob += np.log(feature_val_prob[fea_val])
                else:
                    laplace_prob = 1 / (self._ck_counter[label] + self._Sj[i])
                    prob += np.log(laplace_prob)
            post_prob[label] = prob
        # print(post_prob)
        prob_list = list(post_prob.items())
        # print(prob_list)
        prob_list.sort(key=lambda v:v[1], reverse=True)
        # print(prob_list)
        return prob_list[0][0]


def main():
    X, y = load_iris(return_X_y=True)
    # X, y = load_digits(return_X_y=True)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, shuffle=True)

    model = NaiveBayesScratch()
    # print(xtrain.shape)
    model.fit(xtrain, ytrain)

    # model.predict(xtest[0])
    n_test = xtest.shape[0]
    n_right = 0
    for i in range(n_test):
        y_pred = model.predict(xtest[i])
        if y_pred == ytest[i]:
            n_right += 1
        else:
            print("该样本真实标签为：{}，但是Scratch模型预测标签为：{}".format(ytest[i], y_pred))
    print("Scratch模型在测试集上的准确率为：{}%".format(n_right * 100 / n_test))


if __name__ == '__main__':
    main()












