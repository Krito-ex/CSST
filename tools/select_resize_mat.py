import os.path
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from PIL import Image
import h5py

temp = 'Train_spectral2'
#temp = 'KAIST_selected_1.3'


if temp == 'Train_spectral':
    # data_root = '/home/root/data1/lvtao/CSST2/CSST/datasets/Train_spectral/'
    # output_dir = '/home/root/data1/lvtao/CSST2/CSST/datasets/Train_spectral_resize/'
    data_root = '/home/root/data1/lvtao/CSST/datasets/Train_spectral/'
    output_dir = '/home/root/data1/lvtao/CSST/datasets/Train_spectral_resize/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filenames = os.listdir(data_root)
    filenames.sort()
    for filename in filenames:
        file_path = os.path.join(data_root, filename)
        img = h5py.File(file_path)['cube']
        img = torch.from_numpy(np.array(img)).cuda()
        img = img[2:30,:,:]
        img = img.permute((2,1,0))
        img.resize_(964, 1024, 28)
        img = img / torch.max(img)
        img = img.cpu().numpy()
        img = img.astype(np.float32)
        save_path = os.path.join(output_dir, filename)
        sio.savemat(save_path, {'HSI': img})
        print(filename)

if temp == 'Train_spectral2':
    # data_root = '/home/root/data1/lvtao/CSST2/CSST/datasets/KAIST_selected_1.3/'
    # output_dir = '/home/root/data1/lvtao/CSST2/CSST/datasets/KAIST_selected_1.3_resize/'
    data_root = '/home/root/data1/lvtao/CSST/datasets/Train_spectral/'
    output_dir = '/home/root/data1/lvtao/CSST/datasets/Train_spectral_resize/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filenames = os.listdir(data_root)
    filenames.sort()
    cube_2 = np.zeros((964, 1024, 28))
    for filename in filenames:
        file_path = os.path.join(data_root, filename)
        #cube = sio.loadmat(file_path)['cube']
        cube = h5py.File(file_path)['cube']

        cube = np.array(cube).transpose([2,1,0])
        for i in range(2,30,1):
            img = cube[:, :, i]
            img = Image.fromarray(img.astype(np.float32))
            img_2 = img.resize((1024,964), Image.ANTIALIAS)
            img_2 = np.array(img_2)
            cube_2[:, :, i-2] = img_2
        plt.imshow(cube_2[:, :, 14])
        plt.show()
        save_path = os.path.join(output_dir, filename)
        sio.savemat(save_path, {'HSI': cube_2})
        print('prossed:', filename)


if temp == 'KAIST_selected_1.3':
    # data_root = '/home/root/data1/lvtao/CSST2/CSST/datasets/KAIST_selected_1.3/'
    # output_dir = '/home/root/data1/lvtao/CSST2/CSST/datasets/KAIST_selected_1.3_resize/'
    data_root = '/home/root/data1/lvtao/CSST/datasets/KAIST_selected_1.3/'
    output_dir = '/home/root/data1/lvtao/CSST/datasets/KAIST_selected_1.3_resize/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filenames = os.listdir(data_root)
    filenames.sort()
    cube_586 = np.zeros((586, 586, 28))
    for filename in filenames:
        file_path = os.path.join(data_root, filename)
        cube = sio.loadmat(file_path)['HSI_crop_926']

        for i in range(28):
            img = cube[:, :, i]
            img = Image.fromarray(img.astype(np.float32))           # to image
            img_586 = img.resize((586, 586), Image.ANTIALIAS)  # resize
            img_586 = np.array(img_586)
            cube_586[:, :, i] = img_586
        plt.imshow(cube_586[:, :, 14])
        plt.show()
        save_path = os.path.join(output_dir, filename)
        sio.savemat(save_path, {'HSI_crop_586': cube_586})
        print('prossed:', filename)