#coding:utf-8

import sys, os
import time
import numpy as np
from mnist_load import load_mnist
from conv_net import ConvNet
import matplotlib.pyplot as plt
from PIL import Image
from optimization import *
start = time.time()


class SimpleConv(ConvNet):
    def __init__(self):
        super(SimpleConv, self).__init__()
        ConvNet.add_conv(self, 30, 1, 5, 5)
        ConvNet.add_batch_normalization(self, 30*24*24)
        ConvNet.add_pooling(self, 2, 2, stride=2)
        ConvNet.add_affine(self, 30*12*12, 200)
        ConvNet.add_batch_normalization(self, 200)
        ConvNet.add_affine(self, 200, 10)
        ConvNet.add_softmax(self)

(x_train, t_train), (x_test, t_test) = load_mnist(load_file="mnist.pkl", flatten=False, one_hot_label=False)

network = SimpleConv()

learning_rate = 0.1
optimizer = SGD(lr=learning_rate)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    paras = network.paras
    optimizer.update(paras, grads)
    loss = network.loss(x_batch, t_batch)
    print(loss)
    train_loss_list.append(loss)

    """if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)"""

elapsed_time = time.time() - start
print("elapsed_time : %f [sec]" % elapsed_time)
plt.plot(train_loss_list)
plt.savefig("loss_list_multi")
plt.show()

print("識別テスト")
#S = input("入力待ち")
img = 1 - np.array(Image.open("images.png").convert('L'))/255
img = img.reshape((28*28,))
print(network.predict(img))

#Weight decay,Drop outはまだ実装されていません。


