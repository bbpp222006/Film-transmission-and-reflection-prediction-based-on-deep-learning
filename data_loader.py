import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import hdf5storage


class MatDataSet(Dataset):
    def __init__(self,path):
        '''
        param：
            signal_path_list: 单个数据的文件位置
            signal_label_list: 单个数据的文件标，其实就是文件名称，str格式
        数据默认在根目录下的dataset中
        '''

        data = hdf5storage.loadmat(path)
        layer_n = data['n']
        reflect = data['reflect']
        self.layer_ns = layer_n.reshape(-1, layer_n.shape[1])  #n*5
        self.reflects = reflect.reshape(-1, reflect.shape[1])  #n*90


    # def get_scaler(self,scaler = None): #如果没输入scaler, 则不会对数据进行归一化. 输入了则进行归一化. 都返回scaler
    #     if not scaler:
    #         self.scaler = StandardScaler()
    #         col_rand_array = np.arange(self.predict.shape[0])
    #         np.random.shuffle(col_rand_array)
    #         col_rand = self.predict[col_rand_array[:int(0.5 * len(self.predict))]]
    #         self.scaler.fit_transform(col_rand.reshape((-1,1)))
    #         plt.hist(self.predict.reshape((-1,1)), bins=256, edgecolor='None', facecolor='blue')
    #         plt.show()
    #     else:
    #         self.scaler = scaler
    #         self.predict = self.scaler.transform(self.predict)
    #
    #     return self.scaler



    def __len__(self):
        return len(self.layer_ns)

    def __getitem__(self, idx):
        assert len(self.layer_ns) == len(self.reflects)
        layer_ns = torch.from_numpy(self.layer_ns[idx])
        reflects = torch.from_numpy(self.reflects[idx])
        return layer_ns.float(), reflects.float()


class MatDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(MatDataLoader, self).__init__(*args, **kwargs)


# whole_dataset = MatDataSet(path="data_set/theta.mat")
# split = int(len(whole_dataset)*0.8)
# train_dataset, val_dataset = torch.utils.data.random_split(whole_dataset, [split, len(whole_dataset)-split])
# train_loader = MatDataLoader(train_dataset, batch_size=10, shuffle=True)
# val_loader = MatDataLoader(val_dataset, batch_size=2, shuffle=True)
#
# a= next(iter(val_loader))
# print(a)