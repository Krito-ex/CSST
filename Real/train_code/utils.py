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
        print(HR_code)
        path1 = os.path.join(path) + HR_code
        data = sio.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['HSI']
        HR_HSI[HR_HSI < 0] = 0
        HR_HSI[HR_HSI > 1] = 1
    return HR_HSI

def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch


def generate_adis_3dmasks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/maskII3.mat')
    mask3d = mask['mask3']       #[28,512,512]
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch


def generate_shift_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch**2,1)
    Phi_s_batch[Phi_s_batch==0] = 1
    return Phi_batch, Phi_s_batch

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
    test_data = np.zeros((len(scene_list), 926, 926, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img_dict = sio.loadmat(scene_path)
        if 'img' in img_dict:
            img = img_dict['img'] / np.max(img_dict['img'])
        if 'data_slice' in img_dict:
            img = img_dict['data_slice'] / np.max(img_dict['data_slice'])
        elif 'HSI' in img_dict:
            img = img_dict['HSI'] / np.max(img_dict['HSI'])
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

def LoadTest2(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((4 * len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img_dict = sio.loadmat(scene_path)
        if 'img' in img_dict:
            img = img_dict['img'] / np.max(img_dict['img'])
        if 'data_slice' in img_dict:
            img = img_dict['data_slice'] / np.max(img_dict['data_slice'])
        elif 'HSI' in img_dict:
            img = img_dict['HSI'] / np.max(img_dict['HSI'])
        for j in range(4):
            m = j//2
            n = j % 2
            test_data[i * 4 + j, :, :, :] = img[256 * m:  256 * m +256, 256 * n:  256 * n +256, :]
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

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
