import os.path
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import cv2
from PIL import Image

'''
show measurements as RGB
'''

temp = 'kaist_selected_meas'

if temp == 'kaist_selected_meas':
    # data_root = '/media/tao/HardDisk/krito/CSST/datasets/KAIST_selected'
    # output_dir = '/media/tao/HardDisk/krito/CSST/datasets/KAIST_selected_show/'
    data_root = '/home/data1/lvtao/CSST/datasets/KAIST_selected_meas_show/'
    output_dir = '/home/data1/lvtao/CSST/datasets/KAIST_selected_meas_RGB/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filenames = os.listdir(data_root)
    filenames.sort()
    for i in range(len(filenames)//3):
        filename1 = filenames[3*i]
        filename2 = filenames[3*i+1]
        filename3 = filenames[3*i+2]
        file_path1 = os.path.join(data_root, filename1)
        file_path2 = os.path.join(data_root, filename2)
        file_path3 = os.path.join(data_root, filename3)
        # with cbook.get_sample_data(file_path1) as image_file:
        #     imgB = plt.imread(image_file, 'grayscale')
        # with cbook.get_sample_data(file_path2) as image_file:
        #     imgG = plt.imread(image_file)
        # with cbook.get_sample_data(file_path2) as image_file:
        #     imgR = plt.imread(image_file)
        imgB = cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE)
        imgG = cv2.imread(file_path2, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(file_path3, cv2.IMREAD_GRAYSCALE)
        img = np.zeros((imgB.shape[0],imgB.shape[1],3))
        img[:,:,0] = imgR / np.max(imgR)
        img[:,:,1] = imgG / np.max(imgG)
        img[:,:,2] = imgB / np.max(imgB)
        plt.imshow(img)
        plt.show()
        plt.imsave(output_dir + filename1.split('_')[0] + '.png', img)
        """
         - (M, N) for grayscale images.
        - (M, N, 3) for RGB images.
        - (M, N, 4) for RGBA images.
        """

    

