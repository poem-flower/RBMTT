import numpy as np
import torch
import scipy.io as scio
from torch.utils.data import Dataset
import dill

train_timestep = 20
D_meas = 2  # 量测维度
D_state = 2  # 目标状态维度


class TrMTTDataset(Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self.true_cart = torch.from_numpy(np.array(scio.loadmat(filepath + 'true_state.mat')['X'])).to(torch.float32)  # 三维矩阵，第一维航迹标号，第二维时刻，第三维量测状态
        self.measure_cart = torch.from_numpy(np.array(scio.loadmat(filepath + 'measurement.mat')['Z_cart'])).to(torch.float32)  # 三维矩阵，第一维航迹标号，第二维时刻，第三维目标状态

    def __len__(self):
        return self.true_cart.shape[0]

    def __getitem__(self, idx):
        meas = self.measure_cart[idx, :]  # 量测
        true = self.true_cart[idx, :, 0:2]  # 真实，只取位置信息，速度信息舍弃
        return meas, true


if __name__ == '__main__':
    train_data = TrMTTDataset('./gen_data/train_data/')
    with open('./train_data.pkl', 'wb') as f:
        dill.dump(train_data, f)
    print('done')
