import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
from ssim_torch import ssim

def prepare_data_cave(path, file_num):
    HR_HSI = np.zeros((((1024,1024,28,file_num))))
    file_list = os.listdir(path)
    # for idx in range(1):
    for idx in range(file_num):
        print(f'loading CAVE {idx}')
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code
        data = sio.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['img_expand'] / 65535.0
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI

def prepare_data_KAIST(path, file_num):
    HR_HSI = np.zeros((((2704,3376,28,file_num))))
    file_list = os.listdir(path)
    for idx in range(file_num):
        print(f'loading KAIST {idx}')
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code
        data = sio.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['HSI']
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI


def LoadMeasurement(path_test_meas):
    img = sio.loadmat(path_test_meas)['simulation_test']
    test_data = img
    test_data = torch.from_numpy(test_data)
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

# no use// preserve splice loading
def shuffle_crop(train_data, batch_size, crop_size=926, argument=True):

    if argument:
        gt_batch = torch.zeros((batch_size, 28, crop_size, crop_size), dtype = torch.float32).cuda()

        temp = np.random.randint(0,1)
        if temp == 0:
            index = np.random.choice(range(len(train_data)), batch_size//2)
            processed_data = torch.zeros((batch_size//2, crop_size, crop_size, 28), dtype=torch.float32).cuda()
            for i in range(batch_size//2):
                img = train_data[index[i]]
                h, w, _ = img.shape
                img = torch.from_numpy(img).cuda()
                x_index = np.random.randint(0, h - crop_size)
                y_index = np.random.randint(0, w - crop_size)
                processed_data[i,:,:,:] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
            processed_data = torch.permute(processed_data,(0,3,1,2)).float()
            gt_batch[:batch_size//2,:,:,:] = arguement_1(processed_data)

        else:
            processed_data = torch.zeros((4, 463, 463, 28), dtype = torch.float32).cuda()
            for i in range(batch_size // 2 ):
                sample_list = np.random.randint(0, len(train_data), 1)
                for j in range(4):
                    img = train_data[sample_list[0]]
                    h, w, _ = img.shape
                    img = torch.from_numpy(img).cuda()
                    x_index = np.random.randint(0, h-crop_size//2)
                    y_index = np.random.randint(0, w-crop_size//2)
                    processed_data[j] = img[x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
                gt_batch1 = torch.permute(processed_data, (0,3,1,2)).cuda()
                gt_batch1 = arguement_2(arguement_1(gt_batch1))
                gt_batch[i,:,:,:] = gt_batch1


        # The other half data use splicing.
        processed_data = torch.zeros((4, 463, 463, 28), dtype=torch.float32).cuda()     #（4,128,128,28）
        for i in range(batch_size - batch_size // 2):                                   # 3
            if batch_size == 1:
                h = 926
                w = 926
            sample_list = np.random.randint(0, len(train_data), 4)         #range(0,28)
            for j in range(4):
                img = train_data[sample_list[j]]
                h, w, _ = img.shape
                img = torch.from_numpy(img).cuda()
                x_index = np.random.randint(0, h-crop_size//2)             #（0,512-128） （0,384）
                y_index = np.random.randint(0, w-crop_size//2)             #（0,512-128） （0,384）
                processed_data[j] = img[x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]  #
            gt_batch_2 = arguement_2(torch.permute(processed_data, (0, 3, 1, 2)).cuda()) # [4,28,128,128]
            gt_batch[batch_size//2+i,:,:,:] = gt_batch_2
        # gt_batch = torch.stack(gt_batch, dim=0)
        torch.cuda.empty_cache()
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        torch.cuda.empty_cache()
        return gt_batch

def arguement_1(x):  #processed_data[i]：（28,256,256）
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(2, 3))  #rotation
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(3,))    # flips
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(2,))    # flips
    return x

def arguement_2(generate_gt):
    c, h, w = 28,926,926
    divid_point_h = 463                                            # Corresponds to h, w and modifies
    divid_point_w = 463                                            # Corresponds to h, w and modifies
    output_img = torch.zeros(c,h,w).cuda()                         # stiching the four parts into a single image
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img

def out_back(gt, out_h, out_w):
    [bs, nC, h, w] = gt.shape
    y = int((h - out_h)/2)
    x = int((w - out_w)/2)
    out_h = int(out_h)
    out_w = int(out_w)
    output = gt[:,:, y:y+out_h, x:x+out_w]
    return output

def map_xy(n):
    map = [-5,-3,-1,0,1,3,5]
    return map[n]

def gen_adis_meas_torch(data_batch, Y2H=True, out_h=256, out_w=256):
    #  data_batch, mask3d_batch:[B, nC, H, W]
    [bs, nC, h, w] = data_batch.shape  #bs,28,926,926
    D_num = 3
    offset1 = 40
    step = 1
    center_x = (w - out_w) // 2   #lower left
    center_y = (h - out_h) // 2
    MEA = torch.zeros((bs, nC, out_h, out_w)).cuda().float()
    MEA_B = torch.zeros((bs, nC, out_h, out_w)).cuda().float()
    MEA_G = torch.zeros((bs, nC, out_h, out_w)).cuda().float()
    MEA_R = torch.zeros((bs, nC, out_h, out_w)).cuda().float()
    D_martrix = torch.FloatTensor([
        [0.0                , 0.0               , 0.0               , 0.004418308880642513, 0.0             , 0.0               , 0.0               ],
        [0.0                , 0.0               , 0.004974092060935021, 0.01227308022400698, 0.004974092060935021, 0.0          , 0.0               ],
        [0.0                , 0.004974092060935021, 0.04476682854841519, 0.11045772201606283, 0.04476682854841519, 0.004974092060935021,0.0             ],
        [0.004418308880642513, 0.01227308022400698, 0.11045772201606283, 0.27254350483600964, 0.11045772201606283, 0.01227308022400698, 0.004418308880642513],
        [0.0                , 0.004974092060935021, 0.04476682854841519, 0.11045772201606283, 0.04476682854841519, 0.004974092060935021,0.0             ],
        [0.0                , 0.0               , 0.004974092060935021, 0.01227308022400698, 0.004974092060935021, 0.0          , 0.0               ],
        [0.0                , 0.0               , 0.0               , 0.004418308880642513, 0.0             , 0.0               , 0.0               ]])
    for i in range(nC):
        offset = offset1 + i * step
        mea0 = torch.squeeze(data_batch[:, i, :, :])
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
        MEA[:, i, :, :] = mea
        MEA_B[:, i, :, :] = MEA[:, i, :, :] * RGB_para('B')[i] * 0.01
        MEA_G[:, i, :, :] = MEA[:, i, :, :] * RGB_para('G')[i] * 0.01
        MEA_R[:, i, :, :] = MEA[:, i, :, :] * RGB_para('R')[i] * 0.01
    # RGB，28
    meas_B = torch.sum(MEA_B, 1, keepdim=False)  # bs, row, col
    meas_G = torch.sum(MEA_G, 1, keepdim=False)  # bs, row, col
    meas_R = torch.sum(MEA_R, 1, keepdim=False)  # bs, row, col

    meas_B = meas_B.unsqueeze(1)
    meas_G = meas_G.unsqueeze(1)
    meas_R = meas_R.unsqueeze(1)
    meas_H = torch.cat((meas_B, meas_G, meas_R), dim=1)


    RGB = torch.cat((RGB_para('B').unsqueeze(0), RGB_para('G').unsqueeze(0), RGB_para('R').unsqueeze(0)), dim=0)
    RGB = RGB.unsqueeze(0).unsqueeze(3)
    RGB = RGB.repeat(bs, 1, 1, 28)   #bs 3 28 28

    if Y2H:
        H_H = meas_H / nC
        return H_H, RGB*0.01
    torch.cuda.empty_cache()
    return meas_H, RGB*0.01


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

def init_meas_adis(gt):
    out_h = 256
    out_w = 256
    input_meas = gen_adis_meas_torch(gt, Y2H=True, out_h=out_h, out_w=out_w)
    return input_meas


def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

def dataparallel(model, ngpus, gpu0=0):
    if ngpus==0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:

            model = model.cuda()
    elif ngpus == 1:
        model = model.cuda()
    return model
