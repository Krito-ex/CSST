import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio

class dataset(tud.Dataset):
    # def __init__(self, opt, CAVE, KAIST):
    def __init__(self, opt, train_data1, train_data2):
        super(dataset, self).__init__()
        self.isTrain = opt.isTrain
        self.size = opt.size       # 256
        self.crop_szie = 586  # 256
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

    # def Shuffle_Crop(self, train_data1, train_data2, train_data3):
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
            x = torch.rot90(x, dims=(1, 2))  # 旋转
        # Random vertical Flip
        for j in range(vFlip):
            x = torch.flip(x, dims=(2,))  # 翻转
        # Random horizontal Flip
        for j in range(hFlip):
            x = torch.flip(x, dims=(1,))  # 翻转
        return x

    def Shuffle_Core(self, train_data):
        crop_size = self.crop_szie
        gt_mea_batch = torch.zeros((28, crop_size, crop_size), dtype=torch.float32)
        # only
        processed_data = torch.zeros((crop_size, crop_size, 28), dtype=torch.float32)
        index2 = np.random.choice(range(len(train_data)), 1)
        img = train_data[index2[0]]
        h, w, _ = img.shape
        img = torch.from_numpy(img)
        x_index = random.randint(0, (h - crop_size)//2)
        x_index = 2*x_index
        y_index = random.randint(0, (w - crop_size)//2)
        y_index = 2*y_index
        processed_data[:, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.permute(processed_data, (2, 0, 1)).float()
        gt_mea_batch[:, :, :] = self.Arguement(processed_data)
        return gt_mea_batch

    def RGB_Para(self, temp=None):
        if temp == 'B':
            para = torch.FloatTensor(
                [41.329, 42.693, 42.571, 42.449, 41.015, 40.574, 38.157, 33.801, 31.558, 29.606, 25.58, 18.842, 15.388, 11.558,
                 8.802, 5.814, 4.458, 2.367, 2.258, 1.261, 1.09, 0.751, 0.565, 0.413, 0.265, 0.246, 0.462, 0.582])
        elif temp == 'G':
            para = torch.FloatTensor(
                [8.8, 10.316, 10.9, 11.989, 14.897, 15.848, 19.707, 21.605, 22.494, 23.652, 27.439, 34.137, 38.332, 42.359, 45.631,
                 47.122, 47.225, 43.701, 43.264, 35.983, 33.943, 27.93, 21.888, 14.946, 7.796, 5.055, 3.428, 3.289])
        elif temp == 'R':
            para = torch.FloatTensor(
                [1.852, 1.716, 1.766, 1.823, 1.922, 1.913, 1.738, 1.605, 1.586, 1.593, 1.712, 2.092, 2.426, 2.93, 3.401,
                 3.6, 3.185, 1.542, 1.419, 4.222, 5.614, 20.869, 32.125, 38.664, 39.5, 38.008, 35.146, 34.333])
        else:
            print('please choose the pattern B, G or R')
        return para

    def map_xy(self, n):
        map = [-5, -3, -1, 0, 1, 3, 5]
        return map[n]


    def init_input_adis(self, gt_mea_batch):
        # 去除了原代码中的参数逻辑（Y2H，mul_mask） data_batch, mask3d_batch:[B, nC, H, W]
        crop_size = self.crop_szie
        [N, H, W] = gt_mea_batch.shape  # 29,256,256  28,586,586
        nC = 28
        gt_batch = gt_mea_batch[0:28,:,:]
        out_h = 256
        out_w = 256

        meas_L = torch.ones((nC, 256, 256)).float()
        for i in range(nC):
            meas_L[i, 0::2, 0::2] *= self.RGB_Para('G')[i] * 0.01
            meas_L[i, 1::2, 0::2] *= self.RGB_Para('B')[i] * 0.01
            meas_L[i, 0::2, 1::2] *= self.RGB_Para('R')[i] * 0.01
            meas_L[i, 1::2, 1::2] *= self.RGB_Para('G')[i] * 0.01


        D_num = 3
        pattern = torch.zeros((nC, 512, 512)).float()
        center_x2 = 256
        center_y2 = 256
        center_x = (W - out_w) // 2
        center_y = (H - out_h) // 2
        MEA = torch.zeros((nC, out_h, out_w), dtype=torch.float32)
        D_martrix = torch.FloatTensor([
            [0.0        , 0.0       , 0.0       , 0.004418308880642513, 0.0     , 0.0       , 0.0       ],
            [0.0        , 0.0       , 0.004974092060935021, 0.01227308022400698, 0.004974092060935021, 0.0      , 0.0       ],
            [0.0        , 0.004974092060935021, 0.04476682854841519, 0.11045772201606283, 0.04476682854841519, 0.004974092060935021, 0.0        ],
            [0.004418308880642513, 0.01227308022400698, 0.11045772201606283, 0.27254350483600964, 0.11045772201606283, 0.01227308022400698, 0.004418308880642513],
            [0.0        , 0.004974092060935021, 0.04476682854841519, 0.11045772201606283, 0.04476682854841519, 0.004974092060935021, 0.0        ],
            [0.0        , 0.0       , 0.004974092060935021, 0.01227308022400698, 0.004974092060935021, 0.0      , 0.0       ],
            [0.0        , 0.0       , 0.0       , 0.004418308880642513, 0.0     , 0.0       , 0.0       ]])
        D_martrix = D_martrix * 1 / 0.27254350483600964
        for i in range(nC):
            offset1 = 40 // 2
            step = 1
            offset = offset1 + (i//2) * step
            mea0 = torch.squeeze(gt_batch[i, :, :])
            mea = torch.zeros((out_h, out_w),dtype=torch.float32)
            for x in range(2 * D_num + 1):
                xx = torch.tensor(x - D_num)
                pattern_x = int(center_x2 + self.map_xy(x) * offset)
                mea1_x = int(center_x + self.map_xy(x) * offset)
                for y in range(2 * D_num + 1):
                    yy = torch.tensor(y - D_num)
                    if (torch.abs(xx) + torch.abs(yy)) > 3:
                        continue
                    mea1_y = int(center_y + self.map_xy(y) * offset)
                    mea1 = mea0[mea1_y:mea1_y + out_h, mea1_x:mea1_x + out_w] * D_martrix[x, y]  # 乘以衍射衰减系数
                    mea = mea + mea1
                    pattern_y = int(center_y2 + self.map_xy(y) * offset)
                    pattern[i, pattern_x, pattern_y] = D_martrix[x, y]

            mea[0::2, 0::2] *= self.RGB_Para('G')[i] * 0.01
            mea[1::2, 0::2] *= self.RGB_Para('B')[i] * 0.01
            mea[0::2, 1::2] *= self.RGB_Para('R')[i] * 0.01
            mea[1::2, 1::2] *= self.RGB_Para('G')[i] * 0.01
            MEA[i, :, :] = mea

        meas_H = torch.sum(MEA, dim=0, keepdim=True)
        meas_H = meas_H / nC 

        gt_batch = gt_batch[:, center_y:(center_y+256), center_x:(center_x+256)]
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

