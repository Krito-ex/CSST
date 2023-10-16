import os.path
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from PIL import Image



temp = 'KAIST_926'

if temp == 'KAIST_926':
    data_root = '/home/data1/lvtao/CSST/datasets/KAIST_selected/'
    filenames = os.listdir(data_root)
    filenames.sort()
    for filename in filenames:
        file_path = os.path.join(data_root, filename)
        img = sio.loadmat(file_path)['HSI_crop_926']



if temp == 'kaist_selected':
    # data_root = '/media/tao/HardDisk/krito/CSST/datasets/KAIST_selected'
    # output_dir = '/media/tao/HardDisk/krito/CSST/datasets/KAIST_selected_show/'
    data_root = '/home/data1/lvtao/CSST/datasets/KAIST_selected/'
    output_dir = '/home/data1/lvtao/CSST/datasets/KAIST_selected_show/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filenames = os.listdir(data_root)
    filenames.sort()
    for filename in filenames:
        file_path = os.path.join(data_root, filename)
        img = sio.loadmat(file_path)['HSI_crop_926']
        # for keys in img:
        #     print(keys)
        out_img = img[:, :, 14]
        out_img2 = img[345:601,345:601,14]
        # plt.imshow(out_img)
        # plt.axis('off')
        # plt.show()
        plt.imsave(output_dir + filename.split('.')[0] + '.png', out_img, cmap='gray')
        plt.imsave(output_dir + filename.split('.')[0] + '_256.png', out_img2, cmap='gray')
        print(np.max(out_img))
    


if temp == 'cave_1024':
    data_root = '/media/tao/HardDisk/krito/CSST/datasets/cave_1024_28'
    output_dir = '/media/tao/HardDisk/krito/CSST/datasets/cave_1024_28_show/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filenames = os.listdir(data_root)
    filenames.sort()
    for filename in filenames:
        file_path = os.path.join(data_root, filename)
        img = sio.loadmat(file_path)['img_expand']
        out_img = img[:, :, 14]
        plt.imshow(out_img)
        plt.axis('off')
        plt.show()
        plt.imsave(output_dir + filename.split('.')[0] + '.png', out_img, cmap='gray')

if temp == 'kaist':
    data_root = '/media/tao/HardDisk/krito/CSST/datasets/KAIST_CVPR2021'
    output_dir = '/media/tao/HardDisk/krito/CSST/datasets/KAIST_CVPR2021_show/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filenames = os.listdir(data_root)
    filenames.sort()
    for filename in filenames:
        file_path = os.path.join(data_root, filename)
        img = sio.loadmat(file_path)['HSI']
        out_img = img[:, :, 14]
        plt.imshow(out_img)
        plt.axis('off')
        plt.show()
        plt.imsave(output_dir + filename.split('.')[0] + '.png', out_img, cmap='gray')

if temp == 'tsa':
    data_root = '/home/data1/lvtao/CSST/datasets/TSA_simu_data/Truth'
    output_dir = '/home/data1/lvtao/CSST/datasets/TSA_simu_data/Truth_show/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filenames = os.listdir(data_root)
    filenames.sort()
    for filename in filenames:
        file_path = os.path.join(data_root, filename)
        img = sio.loadmat(file_path)['img']
        out_img = img[:, :, 14]
        plt.imshow(out_img)
        plt.axis('off')
        plt.show()
        plt.imsave(output_dir + filename.split('.')[0] + '.png', out_img, cmap='gray')
