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
start = time.time()
from faces.load_face import load_image
from under_treatment import *

#1.networkの構造を宣言
#conv(3ch -> 64ch) >> BN >> Relu >> pooling >> conv(64ch -> 32ch) >> ... >> softmax
class SimpleConv(ConvNet):
    def __init__(self):
        super(SimpleConv, self).__init__()
        ConvNet.add_conv(self, 3, 64, 3, 3, pad=1)
        ConvNet.add_batch_normalization(self, 64*96*96, "Relu")
        ConvNet.add_pooling(self, 2, 2, stride=2)
        ConvNet.add_conv(self, 64, 32, 3, 3, pad = 1)
        ConvNet.add_batch_normalization(self, 32*48*48, "Relu")
        ConvNet.add_pooling(self, 2, 2, stride=2)
        ConvNet.add_conv(self, 32, 16, 3, 3, pad = 1)
        ConvNet.add_batch_normalization(self, 16*24*24, "Relu")
        ConvNet.add_pooling(self, 2, 2, stride=2)
        ConvNet.add_affine(self, 16*12*12, 256)
        ConvNet.add_batch_normalization(self, 256, "Relu")
        ConvNet.add_affine(self, 256, 6)
        ConvNet.add_softmax(self)
         
def load_face():
    if os.path.exists("./faces/x_train.npy") and os.path.exists("./faces/t_train.npy"):
        x_train = np.load("faces/x_train.npy")
        t_train = np.load("faces/t_train.npy")
        return x_train, t_train
    else:
        dir_list = ["solty_boy", "solty_girl", "soy_boy", "soy_girl", "source_boy", "source_girl"]
        t = []
        x_train = np.empty((0, 3, 96, 96), float)
        for dir_name in dir_list:
            x = load_image("faces/" + dir_name)
            t.append(len(x))
            x_train = np.vstack((x_train, x))
        t_train = np.zeros((len(x_train), len(dir_list)))
        t_mask = np.zeros(len(x_train))
        for i in range(1, len(dir_list)):
            t_mask[np.sum(t[:i]):] = i
        x_train = x_train.astype(np.uint8)
        t_train = np.eye(len(dir_list))[t_mask.tolist()]
        np.save('faces/x_train.npy', x_train)
        np.save('faces/t_train.npy', t_train)
        return x_train, t_train
    
def boost_input(x, t):
    x = np.vstack((x, rolling(x)))
    t = np.vstack((t, t))
    x = np.vstack((x, flipping(x)))
    t = np.vstack((t, t))
    x = np.vstack((x, cropping(x, move_rate=8)))
    t = np.vstack((t, t))
    #x = gcn(x)
    #zca = ZCAWhitening()
    #zca.fit(x)
    #x = zca.transform(x)
    return x, t


#2.学習データの準備
x_train, t_train = load_face()
x_train, t_train = boost_input(x_train, t_train)

print(x_train.shape, t_train.shape)
batch_mask = np.random.choice(len(x_train), 99)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
np.save('faces/x_train_batch.npy', x_batch)
np.save('faces/t_train_batch.npy', t_batch)

#3.networkの生成
network = SimpleConv()

#4.最適化手法の宣言
optimizer = Adam()

#5.input_dataの定義
#本当はここでtrain_dataとtest_dataを分ける。
#今回は分けていません。
input_data = {"x_train": x_train, "t_train": t_train, "x_test": x_train, "t_test": t_train}

#6.trainerの生成
trainer = Trainer(network)

#7.学習
train_loss_list, test_acc_list = trainer.train(optimizer, input_data, 20, batch_size = 100, save="network_online_one")

elapsed_time = time.time() - start
print("elapsed_time : %f [sec]" % elapsed_time)
plt.plot(train_loss_list)
plt.savefig("result/loss_list_multi_face")

plt.figure()
plt.plot(test_acc_list)
plt.savefig("result/test_acc_list_multi_face")
