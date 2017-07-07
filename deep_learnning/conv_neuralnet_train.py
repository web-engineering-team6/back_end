#coding:utf-8

import sys, os
import time
import numpy as np
from mnist_load import load_mnist
from conv_net import ConvNet
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from optimization import *
import pickle
start = time.time()


class SimpleConv(ConvNet):
    def __init__(self):
        super(SimpleConv, self).__init__()
        ConvNet.add_conv(self, 1, 30, 5, 5)
        ConvNet.add_batch_normalization(self, 30*24*24, "Relu")
        ConvNet.add_pooling(self, 2, 2, stride=2)
        ConvNet.add_affine(self, 30*12*12, 200)
        ConvNet.add_batch_normalization(self, 200, "Relu")
        ConvNet.add_affine(self, 200, 10)
        ConvNet.add_softmax(self)

(x_train, t_train), (x_test, t_test) = load_mnist(load_file="mnist.pkl", flatten=False, one_hot_label=False)

network = SimpleConv()

optimizer = Adam()

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
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
    sys.stdout.write("\r%f" % loss)
    sys.stdout.flush()
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        batch_mask = np.random.choice(10000, 1000)
        x_batch = x_test[batch_mask]
        t_batch = t_test[batch_mask]
        test_acc = network.accuracy(x_batch, t_batch)
        test_acc_list.append(test_acc)
        print()
        print('iter%i loss : %f' %(i, loss))
        print('iter%i accuracy : %f' %(i, test_acc))     

elapsed_time = time.time() - start
print("elapsed_time : %i [min]" % (elapsed_time//60))
plt.plot(train_loss_list)
plt.savefig("result/loss_list_multi")

plt.figure()
plt.plot(test_acc_list)
plt.savefig("result/test_acc_list_multi")

with open('network/mnist_network.pkl', mode='wb') as f:
    pickle.dump(network, f)

