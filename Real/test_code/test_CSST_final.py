import os
import argparse
from architecture import *
from utils import *
import torch
import scipy.io as scio
import numpy as np
from torch.autograd import Variable
from utils import dataparallel
import math

'''
Krito/CSST 2023/02/10
real-test
'''

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

parser = argparse.ArgumentParser(description="PyTorch HSIFUSION")
parser.add_argument('--template', default='CSST_5stg',
                    help='You can set various templates in option.py')
parser.add_argument('--data_path', default='./test/testX2/', type=str,help='path of data')
parser.add_argument("--size", default=256, type=int, help='the size of trainset image')
parser.add_argument("--trainset_num", default=2000, type=int, help='total number of trainset')
parser.add_argument("--testset_num", default=3933, type=int, help='total number of testset')
# parser.add_argument("--seed", default=2, type=int, help='Random_seed')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=8, type=int, help='batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='train or test')
parser.add_argument("--pretrained_model_path", default= './model/CSST-5stg/model_epoch_260.pth', type=str)
# parser.add_argument("--pretrained_model_path", default= 'home/root/data1/lvtao/CSST/Real/train_code_adis_CSST-final/exp/CSST_5stg/2023_03_07_02_36_46/model//model_epoch_1.pth', type=str)
parser.add_argument('--method', type=str, default='CSST_5stg', help='method name')
opt = parser.parse_args()
print(opt)

test_num = 3933
save_path = './result/CSST_5stgX2-3/'
def prepare_data(path, file_num):
    HR_HSI = np.zeros((((file_num,1,256,256))))
    for idx in range(file_num):
        path1 = os.path.join(path) + 'scene' + str(idx+1) + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[idx,0,:,:] = data['meas_real']
        HR_HSI[HR_HSI < 0] = 0.0
        HR_HSI[HR_HSI > 1] = 1.0
    return HR_HSI

def prepare_RGB_para():
    meas_L = torch.ones((24, 256, 256)).float()
    for i in range(24):
        meas_L[i, 0::2, 0::2] *= RGB_para('G')[i] * 0.01
        meas_L[i, 1::2, 0::2] *= RGB_para('B')[i] * 0.01
        meas_L[i, 0::2, 1::2] *= RGB_para('R')[i] * 0.01
        meas_L[i, 1::2, 1::2] *= RGB_para('G')[i] * 0.01
    return meas_L

def RGB_para(temp=None):
    if temp == 'B':
        para = np.array(
            [40.9979, 42.6749, 41.9576, 39.9636, 37.4797, 31.6009, 26.2602, 19.4539, 13.2725, 8.7476, 6.0121,
             4.0098, 2.5701, 1.7058, 1.1974, 0.8726, 0.6502, 0.4908, 0.3669, 0.2742, 0.2414, 0.2928, 0.4220, 0.6000])
    elif temp == 'G':
        para = np.array(
            [8.5989, 10.2562, 13.0358, 17.1364, 20.1784, 22.4738, 26.8152, 33.2813, 40.3356, 45.6621, 47.0572, 46.9885,
             44.4106, 40.3525, 35.2413, 30.6469, 24.9280, 18.8221, 12.4761, 8.2776, 5.5343, 4.1256, 3.5096, 3.2794])
    elif temp == 'R':
        para = np.array(
            [1.9095, 1.7142, 1.8797, 1.8781, 1.7006, 1.5861, 1.6856, 2.0386, 2.6779, 3.4102, 3.6246, 2.9502,
             1.7769, 2.0893, 4.5497, 12.1725, 27.5484, 36.1120, 39.5772, 39.6438, 38.3963, 36.8560, 35.4549, 34.2123])
    else:
        print('please choose the pattern B, G or R')
    return para

def prepare_pattern():
    pattern = np.zeros((24, 768, 768), dtype=np.float32)
    center_x2 = 384
    center_y2 = 384
    D_num = 3
    D_martrix = np.array([
        [0.0, 0.0, 0.0, 0.004418308880642513, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.004974092060935021, 0.01227308022400698, 0.004974092060935021, 0.0, 0.0],
        [0.0, 0.004974092060935021, 0.04476682854841519, 0.11045772201606283, 0.04476682854841519, 0.004974092060935021,
         0.0],
        [0.004418308880642513, 0.01227308022400698, 0.11045772201606283, 0.27254350483600964, 0.11045772201606283,
         0.01227308022400698, 0.004418308880642513],
        [0.0, 0.004974092060935021, 0.04476682854841519, 0.11045772201606283, 0.04476682854841519, 0.004974092060935021,
         0.0],
        [0.0, 0.0, 0.004974092060935021, 0.01227308022400698, 0.004974092060935021, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.004418308880642513, 0.0, 0.0, 0.0]])
    for i in range(24):
        offset1 = 51.791
        step = (74.809 - 51.791) // 23  # 来自标定数据，需要仔细确认
        offset = math.ceil(offset1 + i * step)
        # offset1 = 51.825
        # step = (74.858 - 51.825) / 23  # 可能已经不需要再重新训练了
        # offset = math.ceil(offset1 + i * step)  # 向上取整
        for x in range(2 * D_num + 1):
            xx = x - D_num
            pattern_x = int(center_x2 + map_xy(x) * offset)
            for y in range(2 * D_num + 1):
                yy = y - D_num
                if (np.abs(xx) + np.abs(yy)) > 3:
                    continue
                pattern_y = int(center_y2 + map_xy(y) * offset)
                pattern[i, pattern_x, pattern_y] = D_martrix[x, y]
    return pattern


input_meas_H = prepare_data(opt.data_path, test_num)
input_meas_L = prepare_RGB_para()
input_pattern = prepare_pattern()

if opt.method=='hdnet':
    model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path).cuda()
else:
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()
# model = torch.load(pretraind_model_path)
model = model.eval()
# model = dataparallel(model, 1)
psnr_total = 0
k = 0
for j in range(test_num):
    with torch.no_grad():
        meas_H = input_meas_H[j,:,:,:]
        meas_H = meas_H * 0.3
        meas_H = torch.FloatTensor(meas_H)
        meas_H = meas_H.unsqueeze(0)

        meas_L = input_meas_L.unsqueeze(0)

        pattern = torch.from_numpy(input_pattern.copy()).unsqueeze(0)

        meas_H, meas_L, pattern  = Variable(meas_H), Variable(meas_L), Variable(pattern)
        meas_H, meas_L, pattern = meas_H.cuda(), meas_L.cuda(), pattern.cuda()
        if opt.method in ['cst_s', 'cst_m', 'cst_l']:
            model_out, _ = model(meas_H, meas_L, pattern)
        elif 'CSST' in opt.method:
            model_out = model(meas_H, meas_L, pattern)
        else:
            model_out = model(meas_H, meas_L, pattern)
        result = model_out
        result = result.clamp(min=0., max=1.)
    k = k+1

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()  # 256 256 24
    save_file = save_path + f'{j}.mat'
    sio.savemat(save_file, {'res': res})
    print(j+1)




