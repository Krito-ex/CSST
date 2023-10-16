import os
from option import opt
from dataset import dataset
import torch.utils.data as tud
from architecture import *
from utils import *
import torch
import scipy.io as scio
import time
import numpy as np
from torch.autograd import Variable
import datetime
import torch.nn.functional as F

'''
Krito/CSST  2023/02/10
real-train
'''


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# dataset
CAVE = LoadTraining(opt.data_path_CAVE)
KAIST = LoadTraining(opt.data_path_KAIST)
print('dataset is loaded')

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# model
if opt.method=='hdnet':
    model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path).cuda()
else:
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()

# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
# mse = torch.nn.MSELoss().cuda()
criterion = torch.nn.L1Loss()

if __name__ == '__main__':
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # pipeline of training
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    for epoch in range(1, opt.max_epoch):
        model.train()
        Dataset = dataset(opt, CAVE, KAIST)
        loader_train = tud.DataLoader(Dataset, num_workers=8, batch_size=opt.batch_size, shuffle=True, pin_memory=False)

        epoch_loss = 0
        start_time = time.time()
        for i, (input_meas_H, input_meas_L, pattern, label) in enumerate(loader_train):  # [bs c h w]
            input_meas_H, input_meas_L, pattern, label = Variable(input_meas_H), Variable(input_meas_L), Variable(pattern), Variable(label)
            input_meas_H, input_meas_L, pattern, label = input_meas_H.cuda(), input_meas_L.cuda(), pattern.cuda(), label.cuda()

            optimizer.zero_grad(set_to_none=True)
            if 'CSST' in opt.method:
                model_out = model(input_meas_H, input_meas_L, pattern)
                loss = criterion(model_out, label)
            elif opt.method in ['cst_s', 'cst_m', 'cst_l']:
                model_out, diff_pred = model(input_meas_H, input_meas_L, pattern)
                loss = criterion(model_out, label)
                diff_gt = torch.mean(torch.abs(model_out.detach() - label), dim=1, keepdim=True)  # [b,1,h,w]
                loss_sparsity = F.mse_loss(diff_gt, diff_pred)
                loss = loss + 2 * loss_sparsity
            else:
                model_out = model(input_meas_H, input_meas_L, pattern)
                loss = criterion(model_out, label)

            if opt.method == 'hdnet':
                fdl_loss = FDL_loss(model_out, label)
                loss = loss + 0.7 * fdl_loss

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        elapsed_time = time.time() - start_time
        # logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".format(epoch, epoch_loss / len(Dataset), elapsed_time))
        logger.info('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss / len(Dataset), elapsed_time))
        print(epoch)
        if epoch%1 ==0:
            # torch.save(model, os.path.join(opt.outf, 'model_%03d.pth' % (epoch + 1)))
            checkpoint(model, epoch, model_path, logger)




