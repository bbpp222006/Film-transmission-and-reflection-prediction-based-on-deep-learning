import numpy as np
import scipy.io as scio
import math
import cmath
import matplotlib.pyplot as plt


samp_num = 100
theta_num = 89
layer_num = 3
h = 1000
lamda = 600

layer_theta = np.ones((samp_num, theta_num, layer_num + 2)) * 90 * math.pi / 180
layer_rand = np.random.random(size=(samp_num, layer_num)) * 2 + 1
layer_one = np.ones((samp_num, 1))
layer_n = np.hstack((layer_one, layer_rand))
layer_n = np.hstack((layer_n, layer_one))


def calt_theta(layer_theta, layer_n):
    for sample_index, single_sample in enumerate(layer_theta):
        for layer_index in range(layer_num + 1):
            single_sample[:, layer_index + 1] = np.arcsin(
                layer_n[sample_index, layer_index] / layer_n[sample_index, layer_index + 1] * np.sin(
                    single_sample[:, layer_index]))
    return layer_theta


def calt_reflect(layer_theta):
    layer_reflect = np.ones((samp_num, theta_num, layer_num + 1))
    for sample_index, sample_reflect in enumerate(layer_reflect):
        for layer_index in range(layer_num + 1):
            theta1 = layer_theta[sample_index, :, layer_index]
            theta2 = layer_theta[sample_index, :, layer_index + 1]
            tem_sin2 = pow(np.sin(theta1 - theta2) / np.sin(theta1 + theta2), 2)
            tem_tan2 = pow(np.tan(theta1 - theta2) / np.tan(theta1 + theta2), 2)
            sample_reflect[:, layer_index] = (tem_sin2 + tem_tan2) / 2
    return layer_reflect


def calt_sum_reflect(layer_reflect, layer_n, layer_theta):
    theta_reflect = np.ones((samp_num, theta_num))
    for sample_index, sample_reflect in enumerate(layer_reflect):
        for sample_theta_index, sample_theta_reflects in enumerate(sample_reflect):
            r = sample_theta_reflects[-1]
            for reflect_index in reversed(range(len(sample_theta_reflects) - 1)):
                r1 = sample_theta_reflects[reflect_index]
                i_theta = complex(0, 4 * math.pi * layer_n[sample_index, reflect_index + 1] * h * math.cos(
                    layer_theta[sample_index, sample_theta_index, reflect_index + 1]) / lamda)
                r = (r1 + r * cmath.exp(i_theta)) / (1 + r1 * r * cmath.exp(i_theta))
            theta_reflect[sample_index, sample_theta_index] = 1-pow(abs(r), 2)
    return theta_reflect


for single_sample in layer_theta:
    single_sample[:, 0] = np.arange(0.1, 90, 90 / theta_num) * math.pi / 180

layer_theta = calt_theta(layer_theta, layer_n)
layer_reflect = calt_reflect(layer_theta)
theta_reflect = calt_sum_reflect(layer_reflect, layer_n, layer_theta)

scio.savemat('theta.mat', {'n': layer_n, 'reflect': theta_reflect})
fig, ax = plt.subplots(1,1)
ax.plot(np.arange(89), theta_reflect[0])
ax.set(title=layer_n[0],ylabel='Reflect', xlabel='theta')
plt.show()
