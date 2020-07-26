import numpy as np
import scipy.io as scio
import math
import cmath
import matplotlib as mpl
import matplotlib.pyplot as plt


# 生成角度-反射率数据
def output(lamda, hs, ns):
    # 示例(550,1,1,1,1.1,1.2,1.3)
    # 可改参数
    assert len(hs)==len(ns)
    theta_num = 90 #仿真取样点
    layer_num = len(hs)
    lamda = lamda  # nm
    heights = hs
    layer_indexs = [1]+ ns+[1]


    # 每一层光的角度
    def calt_theta(layer_indexs, single_samples):
        single_sample = single_samples
        for layer_index in range(layer_num + 1):
            single_sample[layer_index + 1] = np.arcsin(
                layer_indexs[layer_index] / layer_indexs[layer_index + 1] * np.sin(
                    single_sample[layer_index]))
        return single_sample

    # 不考虑相干情况下的反射系数
    def calt_reflect(layer_theta, sp):
        sample_reflect = np.zeros(layer_num+1)
        for layer_index in range(layer_num + 1):
            theta1 = layer_theta[layer_index]
            theta2 = layer_theta[layer_index + 1]
            if sp:
                _ = np.sin(theta1 - theta2) / np.sin(theta1 + theta2)
            else:
                _ = np.tan(theta1 - theta2) / np.tan(theta1 + theta2)
            sample_reflect[layer_index] = _
        return sample_reflect

    # 考虑相干，并转为反射率
    def calt_sum_reflect(layer_reflect, layer_n, layer_theta):
        sample_theta_reflects = layer_reflect
        r = sample_theta_reflects[-1]
        for reflect_index in reversed(range(len(sample_theta_reflects) - 1)):
            r1 = sample_theta_reflects[reflect_index]
            i_theta = complex(0, 4 * math.pi * layer_n[reflect_index + 1] * heights[reflect_index] * math.cos(
                layer_theta[reflect_index + 1]) / lamda)
            r = (r1 + r * cmath.exp(i_theta)) / (1 + r1 * r * cmath.exp(i_theta))
            theta_reflect = pow(abs(r), 2)
        return theta_reflect

    # 数据生成
    ress = []
    for i in np.arange(1, theta_num):
        single_samples = [i * 2 * math.pi / 360]+(layer_num+1)*[0]
        thetas = calt_theta(layer_indexs, single_samples)
        rs = calt_reflect(thetas, 1)
        rp = calt_reflect(thetas, 0)
        res = calt_sum_reflect(rs, layer_indexs, thetas)
        rep = calt_sum_reflect(rp, layer_indexs, thetas)
        ress.append((res+rep)/2)
    return (ress)

# 可选参数
lamda = 600  #nm
hs = [1000, 1000 ]  #nm
ns = [2.7, 1.9]
testdata = output(lamda, hs, ns)


plt.plot(np.arange(1, 90), testdata)
plt.show()