# coding=utf-8
import torch.autograd
import torch.nn as nn
import os
from data_loader import *
from cgan import *






# 超参数定义
num_epoch = 10
z_dim = 100 # 隐变量维度
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 构造数据集
whole_dataset = MatDataSet(path="data_set/theta.mat")
split = int(len(whole_dataset)*0.8)
train_dataset, val_dataset = torch.utils.data.random_split(whole_dataset, [split, len(whole_dataset)-split])
train_loader = MatDataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = MatDataLoader(val_dataset, batch_size=2, shuffle=True)


# 初始化网络模型
D = discriminator(n_size=5, reflect_size=90).to(device)
G = generator(z_dim=z_dim, n_size=5, reflect_size=90).to(device)
# D = D.to(device)
# G = G.to(device)
D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
D_criterion = nn.BCELoss().to(device)  # 分辨器的是单目标二分类交叉熵函数
G_criterion = nn.BCELoss().to(device) # 折射率的损失函数
print('---------- Networks architecture -------------')
utils.print_network(G)
utils.print_network(D)
print('-----------------------------------------------')

# 载入模型
# G.load_state_dict(torch.load('./generator_CGAN_z100.pth'))
# D.load_state_dict(torch.load('./discriminator_CGAN_z100.pth'))
#########判别器训练train#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    for i, (img, label) in enumerate(train_loader): # img代表折射率，label代表角谱
        num_img = img.size(0)
        img = img.to(device)
        label = label.to(device)
        real_label,fake_label = torch.ones((num_img,1)).to(device),torch.zeros((num_img,1)).to(device) # real_label,fake_label指01

        # print(img.shape)
        # 计算真实图片的损失
        real_out = D(img,label)  # 将真实图片放入判别器中,input,label
        d_loss_real = D_criterion(real_out, real_label)  # 得到真实图片的loss
        # 计算假的图片的损失
        z = torch.randn(num_img, z_dim).to(device)  # 随机生成一些噪声
        __, temp_label = next(iter(train_loader))#挑一张角谱作为条件（懒的写生成了。。）future：更改为新数据
        temp_label = temp_label.to(device)
        fake_img = G(z,temp_label)  # 随机噪声放入生成网络中，生成一张假的图片
        fake_out = D(fake_img,temp_label)  # 判别器判断假的图片
        d_loss_fake = D_criterion(fake_out, fake_label)  # 得到假的图片的loss

        # 损失函数和优化
        d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
        D_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        D_optimizer.step()  # 更新参数

        # ==================训练生成器============================
        ################################生成网络的训练###############################
        z = torch.randn(num_img, z_dim).to(device) # 得到随机噪声
        __, temp_label = next(iter(train_loader))
        temp_label = temp_label.to(device)
        fake_img = G(z,temp_label)
        output = D(fake_img,temp_label)
        g_loss = G_criterion(output, real_label)
        # bp and optimize
        G_optimizer.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        G_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数
        # 打印中间的损失
        # try:
        if (i + 1) % 10 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '.format(
                epoch, num_epoch, d_loss.item(), g_loss.item(),
            ))
        # except BaseException as e:
        #     pass

# 保存模型
torch.save(G.state_dict(), './model/generator_CGAN_z100.pth')
torch.save(D.state_dict(), './model/discriminator_CGAN_z100.pth')

