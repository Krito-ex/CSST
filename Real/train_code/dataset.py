import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio
import math


class dataset(tud.Dataset):
    # def __init__(self, opt, CAVE, KAIST):
    def __init__(self, opt, train_data1, train_data2):
        super(dataset, self).__init__()
        self.isTrain = opt.isTrain
        self.size = opt.size1       # 1006
        self.crop_szie = opt.size2  # 256
        self.batch_size = opt.batch_size
        self.train_data1 = train_data1
        self.train_data2 = train_data2
        # self.train_data3 = train_data3
        if self.isTrain == True:
            self.num = opt.trainset_num
            self.arguement = True
        else:
            self.num = opt.testset_num
            self.arguement = False


    def Shuffle_Crop(self, train_data1, train_data2):
        train_data_index = np.random.choice(range(2), 1)
        if train_data_index == 0:
            gt_mea_batch = self.Shuffle_Core(train_data1)
        elif train_data_index == 1 :
            gt_mea_batch = self.Shuffle_Core(train_data2)
        return gt_mea_batch


    def Arguement(slef, x):  # processed_data[i]：（28,256,256）
        """
        :param x: c,h,w
        :return: c,h,w
        """
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        # Random rotation
        for j in range(rotTimes):
            # x = torch.rot90(x, dims=(1, 2))  # rotation
            x = np.rot90(x, axes=(1,2))
        # Random vertical Flip
        for j in range(vFlip):
            x = np.flip(x, axis=(2,))  # flips
        # Random horizontal Flip
        for j in range(hFlip):
            x = np.flip(x, axis=(1,))  # flips
        return x

    def Shuffle_Core(self, train_data):
        #train_data  256,256,28
        # crop_size = 1006
        crop_size = 256
        gt_mea_batch = np.zeros((25, crop_size, crop_size), dtype=np.float32)
        # only
        processed_data = np.zeros((crop_size, crop_size, 25), dtype=np.float32)
        index1 = np.random.choice(range(len(train_data)), 1)
        img = train_data[index1[0]]
        h, w, _ = img.shape
        x_index = random.randint(0, (h - crop_size)//2)
        x_index = 2*x_index
        y_index = random.randint(0, (w - crop_size)//2)
        y_index = 2*y_index
        processed_data[:, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = np.transpose(processed_data, (2, 0, 1))
        gt_mea_batch[:, :, :] = self.Arguement(processed_data)
        return gt_mea_batch

    def RGB_Para(self, temp=None):
        para = None
        # simulation
        # if temp == 'B':
        #     para = torch.FloatTensor(
        #         [41.329, 42.693, 42.571, 42.449, 41.015, 40.574, 38.157, 33.801, 31.558, 29.606, 25.58, 18.842, 15.388, 11.558,
        #          8.802, 5.814, 4.458, 2.367, 2.258, 1.261, 1.09, 0.751, 0.565, 0.413, 0.265, 0.246, 0.462, 0.582])
        # elif temp == 'G':
        #     para = torch.FloatTensor(
        #         [8.8, 10.316, 10.9, 11.989, 14.897, 15.848, 19.707, 21.605, 22.494, 23.652, 27.439, 34.137, 38.332, 42.359, 45.631,
        #          47.122, 47.225, 43.701, 43.264, 35.983, 33.943, 27.93, 21.888, 14.946, 7.796, 5.055, 3.428, 3.289])
        # elif temp == 'R':
        #     para = torch.FloatTensor(
        #         [1.852, 1.716, 1.766, 1.823, 1.922, 1.913, 1.738, 1.605, 1.586, 1.593, 1.712, 2.092, 2.426, 2.93, 3.401,
        #          3.6, 3.185, 1.542, 1.419, 4.222, 5.614, 20.869, 32.125, 38.664, 39.5, 38.008, 35.146, 34.333])

        # real
        if temp == 'B':
            para = np.array(
                [40.9979, 42.6749, 41.9576, 39.9636, 37.4797, 31.6009, 26.2602, 19.4539, 13.2725, 8.7476, 6.0121,
                 4.0098, 2.5701, 1.7058, 1.1974, 0.8726, 0.6502, 0.4908, 0.3669, 0.2742, 0.2414, 0.2928, 0.4220,0.6000])
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

    def map_xy(self, n):
        map = [-5, -3, -1, 0, 1, 3, 5]
        return map[n]


    def init_input_adis(self, gt_batch):
        # data_batch, mask3d_batch:[B, nC, H, W]
        crop_size = self.crop_szie
        [nC, H, W] = gt_batch[:, :, :].shape  # 24,996,996
        out_w = 256
        out_h = 256

        meas_L = torch.ones((nC-1, 256, 256)).float()
        for i in range(nC-1):
            meas_L[i, 0::2, 0::2] *= self.RGB_Para('G')[i] * 0.01
            meas_L[i, 1::2, 0::2] *= self.RGB_Para('B')[i] * 0.01
            meas_L[i, 0::2, 1::2] *= self.RGB_Para('R')[i] * 0.01
            meas_L[i, 1::2, 1::2] *= self.RGB_Para('G')[i] * 0.01

        D_num = 3
        # matrix_xy = 2 * D_num + 1
        # pattern = torch.zeros((nC,matrix_xy,matrix_xy))
        pattern = np.zeros((nC-1, 768, 768), dtype=np.float32)
        center_x2 = 384
        center_y2 = 384
        center_x = (W - out_w) // 2  # lower-left
        center_y = (H - out_h) // 2
        MEA = np.zeros((nC, out_h, out_w), dtype=np.float32)
        D_martrix = np.array([
            [0.0        , 0.0       , 0.0       , 0.004418308880642513, 0.0     , 0.0       , 0.0       ],
            [0.0        , 0.0       , 0.004974092060935021, 0.01227308022400698, 0.004974092060935021, 0.0      , 0.0       ],
            [0.0        , 0.004974092060935021, 0.04476682854841519, 0.11045772201606283, 0.04476682854841519, 0.004974092060935021, 0.0        ],
            [0.004418308880642513, 0.01227308022400698, 0.11045772201606283, 0.27254350483600964, 0.11045772201606283, 0.01227308022400698, 0.004418308880642513],
            [0.0        , 0.004974092060935021, 0.04476682854841519, 0.11045772201606283, 0.04476682854841519, 0.004974092060935021, 0.0        ],
            [0.0        , 0.0       , 0.004974092060935021, 0.01227308022400698, 0.004974092060935021, 0.0      , 0.0       ],
            [0.0        , 0.0       , 0.0       , 0.004418308880642513, 0.0     , 0.0       , 0.0       ]])
        for i in range(nC-1):
            '''ICCV'''
            # offset1 = 51.791
            # step = (74.809-51.791)//23
            '''ICCV rebuttal'''
            offset1 = 51.825
            step = (74.858-51.825)/23
            offset = math.ceil(offset1 + i * step)   # round up
            for x in range(2 * D_num + 1):
                xx = x - D_num
                pattern_x = int(center_x2 + self.map_xy(x) * offset)
                for y in range(2 * D_num + 1):
                    yy = y - D_num
                    if (np.abs(xx) + np.abs(yy)) > 3:
                        continue
                    pattern_y = int(center_y2 + self.map_xy(y) * offset)
                    pattern[i, pattern_x, pattern_y] = D_martrix[x, y]
        meas_H = gt_batch[24:,:,:]
        gt_batch = gt_batch[0:24,:,:]

        meas_H = torch.FloatTensor(meas_H.copy())
        pattern = torch.FloatTensor(pattern.copy())
        gt_batch = torch.FloatTensor(gt_batch.copy())
        return meas_H, meas_L, pattern, gt_batch

    def __getitem__(self, index):
        if self.isTrain == False:
            print('error, train code here')
        if self.isTrain == True:
            train_data1 = self.train_data1
            train_data2 = self.train_data2

            gt_mea_batch = self.Shuffle_Crop(train_data1, train_data2)  # (29,256,256)
            input_meas_H1, input_meas_L, pattern, GT = self.init_input_adis(gt_mea_batch)
        return input_meas_H1, input_meas_L, pattern, GT

    def __len__(self):
        return self.num   #1250 or 5000

