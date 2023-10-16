import os.path
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from PIL import Image
import math

def LoadTraining(path,scene, i):
    scene_path = path + scene
    if 'mat' not in scene_path:
        print('error in path')
    img_dict = sio.loadmat(scene_path)
    if "img_expand" in img_dict:
        img = img_dict['img_expand'] / 65536.
    if "img" in img_dict:
        img = img_dict['img'] / 65536.
    if "HSI" in img_dict:
        img = img_dict['HSI'] / np.max(img_dict['HSI'])
    if "data_slice" in img_dict:
        img = img_dict['data_slice'] / 65536
    if 'HSI_crop_926' in img_dict:
        img = img_dict['HSI_crop_926'] / np.max(img_dict['HSI_crop_926'])
    elif 'HSI_crop_586' in img_dict:
        img = img_dict['HSI_crop_586'] / np.max(img_dict['HSI_crop_586'])
    # img = img.astype(np.float32)
    img = np.float32(img)
    print('Sence {} is loaded. {}'.format(i, scene))
    return img

def map_xy(n):
    map = [-5,-3,-1,0,1,3,5]
    return map[n]

#寻迹查找法，在926×926中查找与256×256叠加的分量
def gen_adis_meas_torch_xunji(data_batch, Y2H=True, out_h=256, out_w=256):
    # 去除了原代码中的参数逻辑（Y2H，mul_mask） data_batch, mask3d_batch:[B, nC, H, W]
    [bs, nC, h, w] = data_batch.shape  #bs,28,926,926
    D_num = 3          #可以仿真的最多的衍射阶次
    offset1 = 40 // 2
    step = 1
    center_x = (w - out_w) // 2   #只有中间是真正位置不变的，所以锚定中间256×256的左下角位置
    center_y = (h - out_h) // 2
    MEA = torch.zeros((bs, nC, out_h, out_w)).cuda().float()
    # MEA_B = torch.zeros((bs, nC, out_h, out_w)).cuda().float()
    # MEA_G = torch.zeros((bs, nC, out_h, out_w)).cuda().float()
    # MEA_R = torch.zeros((bs, nC, out_h, out_w)).cuda().float()
    # m1 = torch.zeros((bs, 1, out_h // 2, out_w // 2)).cuda().float()
    # m2 = torch.zeros((bs, 1, out_h // 2, out_w // 2)).cuda().float()
    # m3 = torch.zeros((bs, 1, out_h // 2, out_w // 2)).cuda().float()
    # m4 = torch.zeros((bs, 1, out_h // 2, out_w // 2)).cuda().float()
    D_martrix = torch.FloatTensor([
        [0.0                , 0.0               , 0.0               , 0.004418308880642513, 0.0             , 0.0               , 0.0               ],
        [0.0                , 0.0               , 0.004974092060935021, 0.01227308022400698, 0.004974092060935021, 0.0          , 0.0               ],
        [0.0                , 0.004974092060935021, 0.04476682854841519, 0.11045772201606283, 0.04476682854841519, 0.004974092060935021,0.0             ],
        [0.004418308880642513, 0.01227308022400698, 0.11045772201606283, 0.27254350483600964, 0.11045772201606283, 0.01227308022400698, 0.004418308880642513],
        [0.0                , 0.004974092060935021, 0.04476682854841519, 0.11045772201606283, 0.04476682854841519, 0.004974092060935021,0.0             ],
        [0.0                , 0.0               , 0.004974092060935021, 0.01227308022400698, 0.004974092060935021, 0.0          , 0.0               ],
        [0.0                , 0.0               , 0.0               , 0.004418308880642513, 0.0             , 0.0               , 0.0               ]])
    for i in range(nC):
        offset1 = 51.825
        step = (74.858 - 51.825) / 23
        offset = math.ceil(offset1 + i * step)
        mea0 = data_batch[:, i, :, :]
        mea = torch.zeros((bs, out_h, out_w)).cuda().float()
        for x in range(2 * D_num + 1):
            xx = torch.tensor(x - D_num)
            mea1_x = int(center_x + map_xy(x) * offset)
            for y in range(2 * D_num + 1):
                yy = torch.tensor(y - D_num)
                if (torch.abs(xx)+torch.abs(yy)) > 3:
                    continue
                mea1_y = int(center_y + map_xy(y) * offset)
                mea1 = mea0[:, mea1_y:mea1_y + out_h, mea1_x:mea1_x + out_w] * D_martrix[x, y]  # 乘以衍射衰减系数
                mea = mea + mea1

        mea[:, 0::2, 0::2] *= RGB_para('G')[i] * 0.01
        mea[:, 1::2, 0::2] *= RGB_para('B')[i] * 0.01
        mea[:, 0::2, 1::2] *= RGB_para('R')[i] * 0.01
        mea[:, 1::2, 1::2] *= RGB_para('G')[i] * 0.01
        MEA[:, i, :, :] = mea
    meas_H = torch.sum(MEA, dim=1, keepdim=False)
    meas_H = meas_H * 1 / 0.27254350483600964 / nC

    QE, bit = 0.96, 2048
    meas_H = meas_H.cpu().numpy()
    meas_H = np.random.binomial((meas_H * bit / QE).astype(int), QE)
    meas_H = np.float32(meas_H) / np.float32(bit)
    meas_H = torch.from_numpy(meas_H).cuda()

    return meas_H

def RGB_para(temp=None):
    #
    # if temp == 'B':
    #     para = torch.FloatTensor(
    #         [41.329, 42.693, 42.571, 42.449, 41.015, 40.574, 38.157, 33.801, 31.558, 29.606, 25.58, 18.842, 15.388,
    #          11.558, 8.802, 5.814, 4.458, 2.367, 2.258, 1.261, 1.09, 0.751, 0.565, 0.413, 0.265, 0.246, 0.462, 0.582])
    # elif temp == 'G':
    #     para = torch.FloatTensor(
    #         [8.8, 10.316, 10.9, 11.989, 14.897, 15.848, 19.707, 21.605, 22.494, 23.652, 27.439, 34.137, 38.332, 42.359,
    #          45.631, 47.122, 47.225, 43.701, 43.264, 35.983, 33.943, 27.93, 21.888, 14.946, 7.796, 5.055, 3.428, 3.289])
    # elif temp == 'R':
    #     para = torch.FloatTensor(
    #         [1.852, 1.716, 1.766, 1.823, 1.922, 1.913, 1.738, 1.605, 1.586, 1.593, 1.712, 2.092, 2.426, 2.93, 3.401,
    #          3.6, 3.185, 1.542, 1.419, 4.222, 5.614, 20.869, 32.125, 38.664, 39.5, 38.008, 35.146, 34.333])
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
        print('please choose the pattern R, G or B')
    return para

def main():
    data_path1 = '/home/root/data1/lvtao/CSST/datasets/cave_1024_28_30/'
    save_root1 = '/home/root/data1/lvtao/datasets/cave_1024_28_30_gm2/'
    if not os.path.exists(save_root1):
        os.mkdir(save_root1)
    data_path2 = '/home/root/data1/lvtao/CSST/datasets/KAIST_non_selected/'
    save_root2 = '/home/root/data1/lvtao/datasets/KAIST_non_selected_gm2/'
    if not os.path.exists(save_root2):
        os.mkdir(save_root2)
    data_path3 = '/home/root/data1/lvtao/CSST/datasets/Train_spectral_resize/'
    save_root3 = '/home/root/data1/lvtao/datasets/Train_spectral_resize_gm2/'
    if not os.path.exists(save_root3):
        os.mkdir(save_root3)
    data_path4 = '/home/root/data1/lvtao/CSST/datasets/KAIST_selected_1.3_resize/'
    save_root4 = '/home/root/data1/lvtao/datasets/KAIST_selected_1.3_resize_gm2/'
    if not os.path.exists(save_root4):
        os.mkdir(save_root4)
    data_path5 = '/home/root/data1/lvtao/CSST/datasets/cave_1024_28/'
    save_root5 = '/home/root/data1/lvtao/datasets/cave_1024_28_rebuttal/'
    if not os.path.exists(save_root5):
        os.mkdir(save_root5)
    data_path6 = '/home/root/data1/lvtao/CSST/datasets/KAIST_CVPR2021/'
    save_root6 = '/home/root/data1/lvtao/datasets/KAIST_CVPR2021_rebuttal/'
    if not os.path.exists(save_root6):
        os.mkdir(save_root6)
    data_path = data_path6
    save_root = save_root6
    crop_size = 1006
    scene_list = os.listdir(data_path)
    scene_list.sort()
    # mat = sio.loadmat(os.path.join(data_path, scene_list[0]))['img_expand']  # 1024 1024 28
    mat = sio.loadmat(os.path.join(data_path, scene_list[0]))['HSI']
    # mat = sio.loadmat(os.path.join(data_path, scene_list[0]))['HSI_crop_586']
    mat = np.array(mat)
    [h, w, nC] = mat.shape
    out_h = h - crop_size + 256
    out_w = w - crop_size + 256
    print('training sences:', len(scene_list))

    for i in range(len(scene_list)):
        meas = torch.zeros([out_h, out_w, 25])
        train_set = LoadTraining(data_path, scene_list[i], i)  # 30(1024, 1024,28)
        train_set = torch.from_numpy(train_set)
        train_set = train_set[:,:,2:26]
        meas[:,:,0:24] = train_set[375:(h-375),375:(w-375),:]
        train_set = train_set.permute(2, 0, 1).unsqueeze(0).cuda()
        mea1 = gen_adis_meas_torch_xunji(train_set, out_h=out_h, out_w=out_w)
        # mea1 = mea1.squeeze().permute(1,2,0)
        mea1 = mea1.permute(1, 2, 0)
        meas[:, :, 24:] = mea1
        meas = meas.cpu().numpy()
        plt.imshow(mea1[:, :, 0 ].cpu().numpy(), cmap='gray')
        plt.show()
        sio.savemat(os.path.join(save_root,scene_list[i]),{'HSI':meas})

if __name__ == '__main__':
    main()