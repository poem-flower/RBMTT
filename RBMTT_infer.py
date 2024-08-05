# 将所有时刻的结果相加取平均
import torch
from RBMTT import RBMTT
import RBMTT as Tr
import scipy.io as scio
import numpy as np
import load_data as ld
from torch.utils.data import DataLoader


D_state = ld.D_state
D_meas = ld.D_meas
len_window = ld.train_timestep - 2


def window(r_input_window):
    r_output = torch.zeros(1, 1, D_meas).to('cuda')
    memory = model.encode(r_input_window, r_input_mask)
    for i in range(len_window - 1):
        y_mask = (Tr.generate_square_subsequent_mask(r_output.size(0)).type(torch.bool)).to('cuda')
        out = model.decode(r_output, memory, y_mask)
        out = out.transpose(0, 1)  # [batch_size, time_step, d_state], 其中batch_size=1
        r0 = model.linear(out[:, -1])  # 只取最后一个时间步
        r_output = torch.cat([r_output, r0.unsqueeze(0)], dim=0)
    torch.cuda.empty_cache()
    return r_output


def one_track():
    r_input = (z - z_pre)[2:, :]  # 前两个时刻直接输出，舍弃
    time_steps = r_input.shape[0]
    nums = torch.zeros(time_steps, 1, D_meas)  # 记录每个时刻需要除以几
    r = torch.zeros(time_steps, 1, D_meas)
    r[0: len_window] = window(r_input[0: len_window, :])
    nums[0: len_window] = torch.ones(len_window, 1, D_meas)
    for t in range(1, time_steps - len_window + 1):
        r[t: t + len_window, :] = r[t: t + len_window, :] + window(r_input[t: t + len_window, :]).to('cpu')
        nums[t: t + len_window, :] = nums[t: t + len_window, :] + torch.ones(len_window, 1, D_meas)
    r = torch.div(r, nums)
    r = torch.cat([torch.zeros(2, 1, D_meas), r.to('cpu')], dim=0)  # 加上前两个时刻
    return r


if __name__ == '__main__':
    test_data = ld.TrMTTDataset('./gen_data/test_data/')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    filepath = './models/Thu Jun 13 18:01:47 2024/epoch1070'
    model = RBMTT(num_encoder_layers=3,
                  num_decoder_layers=3,
                  emb_size=512,
                  nhead=8,
                  dim_input=D_meas,
                  dim_output=D_state,
                  dim_feedforward=512,
                  dropout=0.1).to('cuda')
    model.load_state_dict(torch.load(filepath + '/model.weights'))
    r_input_mask = (torch.zeros(len_window, len_window)).type(torch.bool).to('cuda')
    for data in test_loader:
        z = data[0].transpose(0, 1).to('cuda')
        z_pre = Tr.data_pre(z)
        r_estimate = one_track()
        r_estimate = r_estimate.detach().numpy()
        track = (z.cpu() - r_estimate).transpose(0, 1).detach().numpy()
        try:
            tracks = np.concatenate((tracks, track), axis=0)
        except:
            tracks = track
        print('one track generated')

    print('all tracks generated')
    scio.savemat('./track.mat', {'tracks': tracks})






