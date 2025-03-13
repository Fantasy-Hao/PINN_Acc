from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(torch.nn.Module):
    """ DNN Class """

    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh

        # layers
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layer_dict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layer_dict)
        self.layers.apply(self.weights_init)

    def forward(self, x):
        out = self.layers(x)
        return out

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.zeros_(m.bias.data)


class modified_MLP(nn.Module):
    """ mMLP Class """

    def __init__(self, layers):
        super(modified_MLP, self).__init__()

        # 创建神经网络层
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        # 初始化编码器权重和偏置
        self.encoder_weights_1 = nn.Parameter(self.xavier_init(layers[1], layers[0]))
        self.encoder_biases_1 = nn.Parameter(torch.zeros(layers[1]))
        self.encoder_weights_2 = nn.Parameter(self.xavier_init(layers[1], layers[0]))
        self.encoder_biases_2 = nn.Parameter(torch.zeros(layers[1]))

        # 权重初始化
        self.apply(self.weights_init)

    def forward(self, H):
        encoder_1 = torch.tanh(F.linear(H, self.encoder_weights_1, self.encoder_biases_1))
        encoder_2 = torch.tanh(F.linear(H, self.encoder_weights_2, self.encoder_biases_2))

        for layer in self.layers[:-1]:
            W = layer.weight
            b = layer.bias
            H = torch.mul(torch.tanh(F.linear(H, W, b)), encoder_1) + \
                torch.mul(1 - torch.tanh(F.linear(H, W, b)), encoder_2)

        H = F.linear(H, self.layers[-1].weight, self.layers[-1].bias)

        return H

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias.data)

    def xavier_init(self, in_dim, out_dim):
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return torch.nn.Parameter(torch.randn(in_dim, out_dim) * xavier_stddev)


"""
DNN 网络的参数总数为 13,052。
modified_MLP 网络的参数总数为 13,452，这比 DNN 网络多400，因为它包含了额外的两个编码器层的参数
"""
