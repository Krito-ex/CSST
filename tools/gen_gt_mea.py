'''
2023/2/17
The training data is pre-processed to generate measurements and saved along the channel dimensions.
No padding is applied to the data
'''

import os.path
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from PIL import Image
import h5py


def LoadTraining(path,scene, i):    # data normalisation
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


def gen_adis_meas_torch(data_batch, Y2H=True, out_h=256, out_w=256):
    # data_batch, mask3d_batch:[B, nC, H, W]
    [bs, nC, h, w] = data_batch.shape  #bs,28,926,926
    D_num = 3                     # preserved diffraction order
    offset1 = 40 // 2
    step = 1
    center_x = (w - out_w) // 2   # lower-left coordinate
    center_y = (h - out_h) // 2
    MEA = torch.zeros((bs, nC, out_h, out_w)).cuda().float()

    D_martrix = torch.FloatTensor([
        [0.0                , 0.0               , 0.0                 ,  0.004418308880642513, 0.0             , 0.0               , 0.0               ],
        [0.0                , 0.0               , 0.004974092060935021,  0.01227308022400698, 0.004974092060935021, 0.0          , 0.0               ],
        [0.0                , 0.004974092060935021, 0.04476682854841519, 0.11045772201606283, 0.04476682854841519, 0.004974092060935021,0.0             ],
        [0.004418308880642513, 0.01227308022400698, 0.11045772201606283, 0.27254350483600964, 0.11045772201606283, 0.01227308022400698, 0.004418308880642513],
        [0.0                , 0.004974092060935021, 0.04476682854841519, 0.11045772201606283, 0.04476682854841519, 0.004974092060935021,0.0             ],
        [0.0                , 0.0               , 0.004974092060935021,  0.01227308022400698, 0.004974092060935021, 0.0          , 0.0               ],
        [0.0                , 0.0               , 0.0               ,    0.004418308880642513, 0.0             , 0.0               , 0.0               ]])
    for i in range(nC):
        offset = offset1 + (i // 2) * step
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
                mea1 = mea0[:, mea1_y:mea1_y + out_h, mea1_x:mea1_x + out_w] * D_martrix[x, y]  # Multiply by the attenuation factor
                mea = mea + mea1

        mea[:, 0::2, 0::2] *= RGB_para('G')[i] * 0.01
        mea[:, 1::2, 0::2] *= RGB_para('B')[i] * 0.01
        mea[:, 0::2, 1::2] *= RGB_para('R')[i] * 0.01
        mea[:, 1::2, 1::2] *= RGB_para('G')[i] * 0.01
        MEA[:, i, :, :] = mea
    meas_H = torch.sum(MEA, dim=1, keepdim=False)

    RGB = torch.cat((RGB_para('B').unsqueeze(0), RGB_para('G').unsqueeze(0), RGB_para('R').unsqueeze(0)), dim=0)
    RGB = RGB.unsqueeze(0).unsqueeze(3)
    RGB = RGB.repeat(bs, 1, 1, 28)   #bs 3 28 28

    if Y2H:
        H_H = meas_H / nC
        return H_H, RGB*0.01
    return meas_H, RGB*0.01

def RGB_para(temp=None):
    para = None
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
    save_root5 = '/home/root/data1/lvtao/datasets/cave_1024_28_gm2/'
    if not os.path.exists(save_root5):
        os.mkdir(save_root5)
    data_path = data_path2
    save_root = save_root2
    crop_size = 586
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
        meas = torch.zeros([out_h, out_w, 29])
        train_set = LoadTraining(data_path, scene_list[i], i)  # 30(1024, 1024,28)
        train_set = torch.from_numpy(train_set)
        meas[:,:,0:28] = train_set[165:(h-165),165:(w-165),:]
        train_set = train_set.permute(2, 0, 1).unsqueeze(0).cuda()   #(1, 28, 1024, 1024)
        mea1, rgb_para1 = gen_adis_meas_torch(train_set, out_h=out_h, out_w=out_w)  # 30  1,3,694, 694
        mea1 = mea1.permute(1, 2, 0)
        meas[:, :, 28:] = mea1
        meas = meas.cpu().numpy()
        plt.imshow(mea1[:, :, 0 ].cpu().numpy(), cmap='gray')
        plt.show()
        sio.savemat(os.path.join(save_root,scene_list[i]),{'HSI':meas})

if __name__ == '__main__':
    main()