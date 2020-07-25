# coding=utf-8
import torch.autograd
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import utils


# 定义判别器  #####Discriminator######使用多层网络来作为判别器
# 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
class discriminator(nn.Module):
    def __init__(self, n_size=5, reflect_size=90):
        super(discriminator, self).__init__()
        self.n_size = n_size
        self.reflect_size = reflect_size

        self.dis = nn.Sequential(
            nn.Linear(n_size+reflect_size, 256),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(256, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.dis(x)
        return x


####### 定义生成器 Generator #####
class generator(nn.Module):
    def __init__(self, z_dim=100, n_size=5, reflect_size=90):
        super(generator, self).__init__()
        self.n_size = n_size
        self.reflect_size = reflect_size
        self.z_dim = z_dim

        self.gen = nn.Sequential(
            nn.Linear(self.z_dim+self.reflect_size, 256),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, n_size),  # 线性变换
            nn.ReLU(True)  # ReLU激活使得生成数据(结构参数n，折射率)分布>0
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.gen(x)
        return x

#
#
# class generator(nn.Module):
#     # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
#     # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
#     def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
#         super(generator, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.input_size = input_size
#         self.class_num = class_num
#
#         self.fc = nn.Sequential(
#             nn.Linear(self.input_dim + self.class_num, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
#             nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
#             nn.ReLU(),
#         )
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
#             nn.Tanh(),
#         )
#         utils.initialize_weights(self)
#
#     def forward(self, input, label):
#         x = torch.cat([input, label], 1)
#         x = self.fc(x)
#         x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
#         x = self.deconv(x)
#
#         return x
#
# class discriminator(nn.Module):
#     # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
#     # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
#     def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
#         super(discriminator, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.input_size = input_size
#         self.class_num = class_num
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 4, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, self.output_dim),
#             nn.Sigmoid(),
#         )
#         utils.initialize_weights(self)
#
#     def forward(self, input, label):
#         x = torch.cat([input, label], 1)
#         x = self.conv(x)
#         x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
#         x = self.fc(x)
#
#         return x
