import os
import torch
import torch.nn as nn
from RBMTT import RBMTT
import RBMTT as RB
import scipy.io as scio
import time
import shutil
import dill
from torch.utils.data import DataLoader
import load_data as ld

device_ids = range(torch.cuda.device_count())
batch_size = 512

D_meas = ld.D_meas
D_state = ld.D_state



if __name__ == '__main__':

    with open('./train_data.pkl', 'rb') as f:
        train_data = dill.load(f)

    train_loader = DataLoader(train_data, batch_size, shuffle=False)
    model = RBMTT(num_encoder_layers=3,
                  num_decoder_layers=3,
                  emb_size=512,
                  nhead=8,
                  dim_input=D_meas,
                  dim_output=D_state,
                  dim_feedforward=512,
                  dropout=0.1).to('cuda')
    model = nn.DataParallel(model, device_ids=device_ids, output_device=0, dim=1)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    L = []
    for epoch in range(10000):
        step = 0
        for data in train_loader:
            meas, true = data[0].transpose(0, 1), data[1].transpose(0, 1)
            meas = meas.to('cuda')  # 量测
            true = true.to('cuda')  # 真实
            meas_pre = RB.data_pre(meas)
            r = (meas - true)[2:, :]  # 真实残差
            src = (meas - meas_pre)[2:, :]  # 前两个时刻直接输出，去掉
            tgt = r[:-1, :]  # decoder输入，去掉结束符
            out = r[1:, :]  # 网络输出，去掉开始符
            r_estimate = model(src, tgt)  # estimate:
            loss = RB.loss_fn(r_estimate, out)
            loss.backward()
            L.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
            step = step+1
            if step % 100 == 0:
                print("Epoch {}, Step {}, Loss: {}".format(epoch, step, loss.item()))

        if epoch % 10 == 0:
            if epoch == 0:
                localtime = time.asctime(time.localtime(time.time()))
                filepath = './models/' + localtime + '/'
                os.makedirs(filepath, exist_ok=True)

            os.makedirs(filepath + 'epoch' + str(epoch), exist_ok=True)
            torch.save(model.module.state_dict(), filepath + 'epoch' + str(epoch) + '/model.weights')
            scio.savemat(filepath + 'epoch' + str(epoch) + '/L.mat', {'Loss': L})

