# coding: utf-8

import pickle
import numpy as np


def _change_ont_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False, load_file="mnist.pkl"):
    with open(load_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


#(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

#img = x_train[0]
#label = t_train[0]
#print(label)  # 5

#print(img.shape)  # (784,)
#img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
#print(img.shape)  # (28, 28)
