import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
from ssim_torch import ssim
from architecture import *
from fvcore.nn import FlopCountAnalysis

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand'] / 65536.
        if "img" in img_dict:
            img = img_dict['img'] / 65536.
        if "HSI" in img_dict:     #cave,kaist,Train_spectral
            img = img_dict['HSI']
        elif "data_slice" in img_dict:
            img = img_dict['data_slice'] / 65536
        elif 'HSI_crop_926' in img_dict:
            img = img_dict['HSI_crop_926']  / 65536
        img = np.array(img).astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    # test_data = np.zeros((len(scene_list), 256, 256, 29))
    test_data = np.zeros((len(scene_list), 586, 586, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img_dict = sio.loadmat(scene_path)
        if 'img' in img_dict:
            img = img_dict['img'] / np.max(img_dict['img'])
        if 'data_slice' in img_dict:
            img = img_dict['data_slice'] / np.max(img_dict['data_slice'])
        if 'HSI' in img_dict:
            img = img_dict['HSI'] / np.max(img_dict['HSI'])
        if 'HSI_crop_926' in img_dict:
            img = img_dict['HSI_crop_926'] / np.max(img_dict['HSI_crop_926'])
        if 'HSI_crop_586' in img_dict:
            img = img_dict['HSI_crop_586'] / np.max(img_dict['HSI_crop_586'])
        elif 'HSI_crop_256' in img_dict:
            img = img_dict['HSI_crop_256'] / np.max(img_dict['HSI_crop_256'])
        test_data[i, :, :, :] = img
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def init_input_adis(test_gt_mea):
    [bs, N, H, W] = test_gt_mea.shape  # 10, 31,256,256
    nC = 28
    out_h = 256
    out_w = 256
    gt_batch = test_gt_mea[:,0:28, :, :]

    meas_L = torch.ones((bs, nC, 256, 256)).cuda().float()
    for i in range(nC):
        meas_L[:, i, 0::2, 0::2] *= RGB_para('G')[i] * 0.01
        meas_L[:, i, 1::2, 0::2] *= RGB_para('B')[i] * 0.01
        meas_L[:, i, 0::2, 1::2] *= RGB_para('R')[i] * 0.01
        meas_L[:, i, 1::2, 1::2] *= RGB_para('G')[i] * 0.01

    D_num = 3
    pattern = torch.zeros((bs, nC, 512, 512)).cuda().float()
    center_x2 = 256
    center_y2 = 256
    center_x = (W - out_w) // 2                        # lower left
    center_y = (H - out_h) // 2                        # lower left
    MEA = torch.zeros((bs, nC, out_h, out_w), dtype=torch.float32).cuda()
    D_martrix = torch.FloatTensor([
        [0.0                 , 0.0                 , 0.0                 , 0.004418308880642513, 0.0                 , 0.0                 , 0.0],
        [0.0                 , 0.0                 , 0.004974092060935021, 0.01227308022400698 , 0.004974092060935021, 0.0                 , 0.0 ],
        [0.0                 , 0.004974092060935021, 0.04476682854841519 , 0.11045772201606283 , 0.04476682854841519 , 0.004974092060935021,0.0],
        [0.004418308880642513, 0.01227308022400698 , 0.11045772201606283 , 0.27254350483600964 , 0.11045772201606283 , 0.01227308022400698 , 0.004418308880642513],
        [0.0                 , 0.004974092060935021, 0.04476682854841519 , 0.11045772201606283 , 0.04476682854841519 , 0.004974092060935021,0.0],
        [0.0                 , 0.0                 , 0.004974092060935021, 0.01227308022400698 , 0.004974092060935021, 0.0                 , 0.0],
        [0.0                 , 0.0                 , 0.0                 , 0.004418308880642513, 0.0                 , 0.0                 , 0.0]])
    D_martrix = D_martrix * 1 / 0.27254350483600964
    for i in range(nC):
        offset1 = 40 // 2
        step = 1
        offset = offset1 + (i //2 ) * step
        mea0 = torch.squeeze(gt_batch[:, i, :, :])
        mea = torch.zeros((bs, out_h, out_w),dtype=torch.float32).cuda()
        for x in range(2 * D_num + 1):
            xx = torch.tensor(x - D_num)
            pattern_x = int(center_x2 + map_xy(x) * offset)
            mea1_x = int(center_x + map_xy(x) * offset)
            for y in range(2 * D_num + 1):
                yy = torch.tensor(y - D_num)
                if (torch.abs(xx) + torch.abs(yy)) > 3:
                    continue
                mea1_y = int(center_y + map_xy(y) * offset)
                mea1 = mea0[:, mea1_y:mea1_y + out_h, mea1_x:mea1_x + out_w] * D_martrix[x, y]
                mea = mea + mea1
                pattern_y = int(center_y2 + map_xy(y) * offset)
                pattern[:,i, pattern_x, pattern_y] = D_martrix[x, y]
        mea[:, 0::2, 0::2] *= RGB_para('G')[i] * 0.01
        mea[:, 1::2, 0::2] *= RGB_para('B')[i] * 0.01
        mea[:, 0::2, 1::2] *= RGB_para('R')[i] * 0.01
        mea[:, 1::2, 1::2] *= RGB_para('G')[i] * 0.01
        MEA[:, i, :, :] = mea
    meas_H = torch.sum(MEA, dim=1, keepdim=True)
    meas_H = meas_H / nC 
    gt_batch = gt_batch[:, :, center_y:(center_y+256), center_x:(center_x+256)]
    return meas_H, meas_L, pattern, gt_batch


def map_xy(n):
    map = [-5,-3,-1,0,1,3,5]
    return map[n]

def RGB_para(temp=None):
    if temp == 'B':
        para = torch.FloatTensor(
            [41.329, 42.693, 42.571, 42.449, 41.015, 40.574, 38.157, 33.801, 31.558, 29.606, 25.58, 18.842, 15.388,
             11.558, 8.802, 5.814, 4.458, 2.367, 2.258, 1.261, 1.09, 0.751, 0.565, 0.413, 0.265, 0.246, 0.462, 0.582])
    elif temp == 'G':
        para = torch.FloatTensor(
            [8.8, 10.316, 10.9, 11.989, 14.897, 15.848, 19.707, 21.605, 22.494, 23.652, 27.439, 34.137, 38.332, 42.359,
             45.631, 47.122, 47.225, 43.701, 43.264, 35.983, 33.943, 27.93, 21.888, 14.946, 7.796, 5.055, 3.428, 3.289])
    elif temp == 'R':
        para = torch.FloatTensor(
            [1.852, 1.716, 1.766, 1.823, 1.922, 1.913, 1.738, 1.605, 1.586, 1.593, 1.712, 2.092, 2.426, 2.93, 3.401,
             3.6, 3.185, 1.542, 1.419, 4.222, 5.614, 20.869, 32.125, 38.664, 39.5, 38.008, 35.146, 34.333])
    else:
        print('please choose the pattern R, G or B')
    return para


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

def my_summary(test_model, H = 256, W = 256, C = 28, N = 1):
    model = test_model.cuda()
    print(model)
    inputs = torch.randn((N, 1, 256, 256)).cuda()
    # y_l = torch.randn((1,3,28,28)).cuda()
    # pattern = torch.randn((N, C, 512, 512)).cuda()
    y_l = torch.randn((1,28,256,256)).cuda()
    pattern = torch.randn((N, C, 512, 512)).cuda()


    # inputs = [inputs, y_l, pattern]
    flops = FlopCountAnalysis(model,(inputs, y_l, pattern))
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')

if __name__ == '__main__':

    # model, FDL_loss = model_generator('tsa_net', '/home/root/data1/lvtao/CSST/D-simulation/train_code_D_ADIS/exp/tsa_net/2023_04_20_02_10_24/model/model_epoch_211.pth')
    model = model_generator('CSST_9stg', '/home/root/data1/lvtao/CSST/D-simulation/train_code_D_ADIS2/exp/CSST_9stg/2023_04_09_09_18_28/model/model_epoch_241.pth')
    my_summary(model)