'''
2022/11/28
Calculation of diffraction order parameters
'''
import numpy as np
import torch
import os

def Cal_decayxy_all(D_num):
    decayxy_all = 0
    for x in range(2 * D_num + 1):
        xx = np.abs(x - D_num)
        if xx == 0:
            decay_x = 1
        else:
            decay_x = 4 / np.square((2 * xx - 1) * np.pi)
        for y in range(2 * D_num + 1):
            yy = np.abs(y - D_num)
            if yy == 0:
                decay_y = 1
            else:
                decay_y = 4 / np.square((2 * yy - 1) * np.pi)
            decay_xy = decay_x * decay_y

            if decay_xy < 0.0144:
                decay_xy = 0
            # print(decay_xy)
            decayxy_all = decayxy_all + decay_xy
            print(decay_xy / 3.6691389897612985)
    return decayxy_all

D_martrix = torch.FloatTensor([
    [0.0                 , 0.0                 , 0.0                 , 0.004418308880642513, 0.0                 , 0.0                 , 0.0                 ],
    [0.0                 , 0.0                 , 0.004974092060935021, 0.01227308022400698 , 0.004974092060935021, 0.0                 , 0.0                 ],
    [0.0                 , 0.004974092060935021, 0.04476682854841519 , 0.11045772201606283 , 0.04476682854841519 , 0.004974092060935021, 0.0                 ],
    [0.004418308880642513, 0.01227308022400698 , 0.11045772201606283 , 0.27254350483600964 , 0.11045772201606283 , 0.01227308022400698 , 0.004418308880642513],
    [0.0                 , 0.004974092060935021, 0.04476682854841519 , 0.11045772201606283 , 0.04476682854841519 , 0.004974092060935021, 0.0                 ],
    [0.0                 , 0.0                 , 0.004974092060935021, 0.01227308022400698 , 0.004974092060935021, 0.0                 , 0.0                 ],
    [0.0                 , 0.0                 , 0.0                 , 0.004418308880642513, 0.0                 , 0.0                 , 0.0                 ]
])


def main():
    D_num = 3
    decayxy_all = Cal_decayxy_all(D_num)
    print(decayxy_all)
    D_martrix_sum = np.sum(D_martrix.numpy())
    print(D_martrix_sum)

if __name__ == '__main__':
    main()