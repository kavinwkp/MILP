from collections import defaultdict
import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

n_sample, n_feature = X.shape
# print(n_sample)     # 150
# print(n_feature)    # 4
ck, ck_num = np.unique(y, return_counts=True)
# print(ck)               # [0 1 2]
# print(ck_num)           # [50 50 50]
ck_counter = dict(zip(ck, ck_num))
# print(ck_counter)       # {0: 50, 1: 50, 2: 50}
prior_prob = defaultdict(float)
for label, label_num in ck_counter.items():
    prior_prob[label] = (label_num + 1) / (n_sample + ck.shape[0])
# print(prior_prob)         # {0: 0.3, 1: 0.3, 2: 0.3}
ck_idx = list()
for label in ck:
    label_idx = np.squeeze(np.argwhere(y == label))
    ck_idx.append(label_idx)
# print(ck_idx)     # [array([ 0, ..., 49]), array([50, ..., 99]), array([100, ..., 149])]
for label, idx in zip(ck, ck_idx):
    xdata = X[idx]
    label_likelihood = defaultdict(defaultdict)
    # for i in range(n_feature):




