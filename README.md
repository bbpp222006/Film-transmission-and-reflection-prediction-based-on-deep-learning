# 基于深度学习的多层薄膜透反射谱的预测以及逆向

## 简介
这是以前的一个项目，最近整理出来留作备份

## 食用指南
首先运行creat_data.py，生成对应的数据集。
> 各种参数可以在内部进行修改，包括折射率范围、数据集多少等等

然后运行main.py对网络进行训练。
训练完成后，使用生成网络即可对特定的角谱进行薄膜结构的逆向设计。（暂时没做封装）

## 文件结构

### 数据生成
creat_data.py 是一个手写的薄膜仿真文件，能够相比于comsol以及fdtd等商业软件更快的生成数据。  
算法参考物理光学梁栓延中的透反射那一章（忘了哪一页了……）  
能够自定义设置层厚度，波长，入射角度的抽样点数，层的个数以及仿真的个数。
![](undefined/mk_pic/2020-07-20-22-23-23.png)

以后可能会更新自定义每一层各自的厚度等功能。

### 网络模型
~~这个暂时没找到……当初用的keras写的，模型和算法文件丢失了……~~
网络结构是一个很简单的类自编码器结构。（以用cgan代替）
>参考文献：D. Liu, Y. Tan, E. Khoram, and Z. Yu, “Training deep neural networks for the inverse design of nanophotonicstructures,” arXiv e-prints arXiv:1710.04724 (2017).
~~以后看需求，会用torch重写补上。~~

******





## 更新计划

- [x] 完成数据加载以及网络编写
- [x] 增加仿真脚本中自定义厚度的功能
- [ ] 优化训练网络内部逻辑
  - [ ] 完成自定义角谱生成，而不是从数据集中抽取
- [ ] 代码优化整理
- [ ] 结果分析

2020/7/25更新

- 修改create_data，之前的是错误版本，现阶段能够正常进行仿真了
- 增加dataloader类
- 增加cgan（条件生成对抗网络）,训练完成后能够输入所需角谱，得到对应的薄膜折射率分布。

2020/7/26更新
新增creat_data_simplify.py文件。
此文件能够自定义每一层的厚度和折射率还有层数。
>注意：厚度列表和折射率列表需要长度相同。