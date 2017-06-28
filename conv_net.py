# coding:utf-8

import sys, os

sys.path.append(os.pardir)
import numpy as np
from layer_class import *
from collections import OrderedDict


class ConvNet:

    def __init__(self):
        self.layer_num = 0
        self.batch_norm_num = 0
        self.paras = {}
        self.layers = OrderedDict()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)
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
        
        return grad

    def add_conv(self, input_chanel, output_chanel, filter_width, filter_depth, stride=1, pad=0):
        self.paras["W" + str(self.layer_num)] = np.sqrt(2 / input_chanel * filter_width * filter_depth)\
                                                * np.random.randn(output_chanel, input_chanel, filter_width, filter_depth)
        self.paras["b" + str(self.layer_num)] = np.zeros(output_chanel)
        self.layers["layer" + str(self.layer_num)] =\
            Convolution(self.paras["W" + str(self.layer_num)], self.paras["b" + str(self.layer_num)], stride, pad)
        self.layer_num += 1

    def add_deconv(self, input_chanel, output_chanel, filter_width, filter_depth, stride=1, pad=0):
        self.paras["W" + str(self.layer_num)] = np.sqrt(2 / input_chanel * filter_width * filter_depth)\
                                                * np.random.randn(output_chanel, input_chanel, filter_width, filter_depth)
        self.paras["b" + str(self.layer_num)] = np.zeros(output_chanel)
        self.layers["layer" + str(self.layer_num)] =\
            Deconvolution(self.paras["W" + str(self.layer_num)], self.paras["b" + str(self.layer_num)], stride, pad)
        self.layer_num += 1

    def add_batch_normalization(self, input_size, function):
        self.paras['gamma' + str(self.batch_norm_num)] = np.ones(input_size)
        self.paras['beta' + str(self.batch_norm_num)] = np.zeros(input_size)
        self.layers['BatchNorm' + str(self.batch_norm_num)] =\
            BatchNormalization(self.paras['gamma' + str(self.batch_norm_num)], self.paras['beta' + str(self.batch_norm_num)])
        self.layers[function + str(self.batch_norm_num)] = eval(function)()
        self.batch_norm_num += 1

    def add_pooling(self, pool_h, pool_w, stride=1, pad=0):
        self.layers["Pooling" + str(self.layer_num)] = Pooling(pool_h, pool_w, stride, pad)

    def add_affine(self, input_size, output_size, output_shape=None):
        self.paras["W" + str(self.layer_num)] = np.sqrt(2 / input_size)\
                                                                   * np.random.randn(input_size, output_size)
        self.paras["b" + str(self.layer_num)] = np.zeros(output_size)
        self.layers["layer" + str(self.layer_num)] =\
            Affine(self.paras["W" + str(self.layer_num)], self.paras["b" + str(self.layer_num)], output_shape=output_shape)
        self.layer_num += 1

    def add_softmax(self):
        self.lastLayer = SoftmaxWithLoss()

    def add_sigmoid(self):
        self.layers["Sigmoid" + str(self.layer_num)] = Sigmoid()




