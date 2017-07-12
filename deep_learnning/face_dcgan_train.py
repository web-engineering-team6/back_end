#coding:utf-8

import sys, os
import time
import numpy as np
from conv_net import ConvNet
from optimization import *
from sys import argv
from dcgan_trainer import DCGAN_trainer
from faces.load_face import load_image
from under_treatment import *

class Generator(ConvNet):
    def __init__(self, nz):
        super(Generator, self).__init__()
        ConvNet.add_affine(self, nz, 512)
        ConvNet.add_batch_normalization(self, 512, "Relu")
        ConvNet.add_affine(self, 512, 64*12*12, output_shape=(16, 12, 12))
        ConvNet.add_batch_normalization(self, 64*12*12, "Relu")
        ConvNet.add_deconv(self, 64, 32, 4, 4, stride=2 ,pad=1, wscale=0.3)
        ConvNet.add_batch_normalization(self, 32*24*32, "Relu")
        ConvNet.add_deconv(self, 32, 16, 4, 4, stride=2, pad=1, wscale=0.3)
        ConvNet.add_batch_normalization(self, 16*48*48, "Relu")
        ConvNet.add_deconv(self, 16, 3, 4, 4, stride=2, pad=1, wscale=0.3)
        ConvNet.add_tanh(self)
        #最後の層をどうするか、disの学習をどのように抑えるか

    def gradient_gen(self, dout):
        layers = list(self.layers.values())
        for layer in reversed(layers):
            dout = layer.backward(dout)

        grad = {}
        for i in range(self.layer_num):
            grad["W" + str(i)] = self.layers["layer" + str(i)].dW
            grad["b" + str(i)] = self.layers["layer" + str(i)].db
        for i in range(self.batch_norm_num):
            grad["gamma" + str(i)] = self.layers["BatchNorm" + str(i)].dgamma
            grad["beta" + str(i)] = self.layers["BatchNorm" + str(i)].dbeta

        return grad


class Discriminator(ConvNet):
    def __init__(self):
        super(Discriminator,self).__init__()
        ConvNet.add_conv(self, 3, 16, 4, 4, stride=2, pad=1)
        ConvNet.add_batch_normalization(self, 16*48*48, "Elu")
        ConvNet.add_conv(self, 16, 32, 4, 4, stride=2, pad=1)
        ConvNet.add_batch_normalization(self, 32*24*24, "Elu")
        ConvNet.add_conv(self, 32, 64, 4, 4, stride=2, pad=1)
        ConvNet.add_batch_normalization(self, 64*12*12, "Elu")
        ConvNet.add_affine(self, 64*12*12, 512)
        ConvNet.add_batch_normalization(self, 512, "Elu")
        ConvNet.add_affine(self, 512, 2)
        ConvNet.add_softmax(self)


    def back_going(self, y, t):
        loss = self.lastLayer.forward(y, t)
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        # layers.reverse()
        for layer in reversed(layers):
            dout = layer.backward(dout)

        grad = {}
        for i in range(self.layer_num):
            grad["W" + str(i)] = self.layers["layer" + str(i)].dW
            grad["b" + str(i)] = self.layers["layer" + str(i)].db
        for i in range(self.batch_norm_num):
            grad["gamma" + str(i)] = self.layers["BatchNorm" + str(i)].dgamma
            grad["beta" + str(i)] = self.layers["BatchNorm" + str(i)].dbeta

        return loss, dout,grad

def load_faces():
	if os.path.exists("./faces/x_train.npy"):
        x_train = np.load("faces/x_train.npy")
        return x_train
    else:
        dir_list = ["solty_boy", "solty_girl", "soy_boy", "soy_girl", "source_boy", "source_girl"]
        t = []
        x_train = np.empty((0, 3, 96, 96), float)
        for dir_name in dir_list:
            x = load_image("faces/" + dir_name)
            x_train = np.vstack((x_train, x))
        x_train = x_train.astype(np.uint8)
        np.save('faces/x_train.npy', x_train)
        return x_train
        

def boost_input(x):
    x = np.vstack((x, rolling(x)))
    x = np.vstack((x, flipping(x)))
    x = np.vstack((x, cropping(x, move_rate=8)))
    return x


nz = 100
batch_size = 100
epoch_num = 20  #10000

try:
    save = argv[1]
except:
    save = input("saveディレクトリを指定してください。")
    
print("コメントを書き込んでください。quitで終了します。")
memo = ''
input_memo = ''
while True:
    input_memo =  input()
    if input_memo == 'quit':
        break
    memo += input_memo + '\n'      

gen = Generator(nz)
dis = Discriminator()

learning_rate_gen = 2e-4
learning_rate_dis = 1e-5
opt_gen = Adam(lr=learning_rate_gen)
opt_dis = Adam(lr=learning_rate_dis)

input_img = load_faces()
input_img = boost_input(input_img)

#学習
dc_trainer = DCGAN_trainer(gen, dis)
dc_trainer.train(opt_gen, opt_dis, input_img, epoch_num, nz=nz, batch_size=batch_size, save="dcgac_faces", img_test="img_test", graph="graph"):


minute = (time.time() - start) // 60
hour = minute // 60
minute = minute % 60
memo += '\n\n%d時間%d分かかりました' % (hour, minute)

with open(save + "/memo", mode='w') as f:
    f.write(memo)