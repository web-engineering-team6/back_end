#coding:utf-8

import sys, os
import time
import numpy as np
from conv_net import ConvNet
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from optimization import *
import pickle
from trainer import Trainer
from faces.load_face import load_image_list
import glob
start = time.time()


class SimpleConv(ConvNet):
    def __init__(self):
        super(SimpleConv, self).__init__()
        ConvNet.add_conv(self, 3, 64, 3, 3, pad=1)
        ConvNet.add_batch_normalization(self, 64*96*96, "Relu")
        ConvNet.add_pooling(self, 2, 2, stride=2)
        ConvNet.add_conv(self, 64, 16, 3, 3, pad = 1)
        ConvNet.add_batch_normalization(self, 16*48*48, "Relu")
        ConvNet.add_pooling(self, 2, 2, stride=2)
        ConvNet.add_affine(self, 16*24*24, 200)
        ConvNet.add_batch_normalization(self, 200, "Relu")
        ConvNet.add_affine(self, 200, 2)
        ConvNet.add_softmax(self)
         
def load_dust():
    if os.path.exists("./faces/x_dust.npy") and os.path.exists("./faces/t_dust.npy"):
        x_train = np.load("faces/x_dust.npy")
        t_train = np.load("faces/t_dust.npy")
        return x_train, t_train
    else:
        dir_list = [glob.glob("faces/miss/*"), \
                    ["faces/solty_boy", "faces/solty_girl", "faces/soy_boy", "faces/soy_girl", "faces/source_boy", "faces/source_girl"]]
        t = []
        x_train = np.empty((0, 3, 96, 96), float)
        for dir_names in dir_list:
            x = load_image_list(dir_names)
            t.append(len(x))
            x_train = np.vstack((x_train, x))
        t_train = np.zeros((len(x_train), len(dir_list)))
        t_mask = np.zeros(len(x_train))
        for i in range(1, len(dir_list)):
            t_mask[np.sum(t[:i]):] = i
        x_train = x_train.astype(np.uint8)
        t_train = np.eye(len(dir_list))[t_mask.tolist()]
        np.save('faces/x_dust.npy', x_train)
        np.save('faces/t_dust.npy', t_train)
        return x_train, t_train

x_train, t_train = load_dust()
x_train = x_train[:len(x_train) - len(x_train) % 100]
t_train = t_train[:len(t_train) - len(t_train) % 100]

print(x_train.shape, t_train.shape)

network = SimpleConv()

optimizer = Adam()

input_data = {"x_train": x_train, "t_train": t_train, "x_test": x_train, "t_test": t_train}

trainer = Trainer(network)
train_loss_list, test_acc_list = trainer.train(optimizer, input_data, 10, batch_size = 100, save="dust_network")

elapsed_time = time.time() - start
print("elapsed_time : %f [sec]" % elapsed_time)
plt.plot(train_loss_list)
plt.savefig("result/loss_list_multi_face")

plt.figure()
plt.plot(test_acc_list)
plt.savefig("result/test_acc_list_multi_face")
