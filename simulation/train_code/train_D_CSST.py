import os
from option import opt
from dataset import dataset
from architecture import *
from utils import *
import torch
import scipy.io as scio
import time
import numpy as np
from torch.autograd import Variable
import datetime
import torch.nn.functional as F
import torch.utils.data as tud

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# dataset
train_set1 = LoadTraining(opt.data_path1)
train_set2 = LoadTraining(opt.data_path2)
test_data = LoadTest(opt.test_path)
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
mse = torch.nn.MSELoss().cuda()

def train(epoch, logger):
    model.train()
    Dataset = dataset(opt, train_set1, train_set2)
    loader_train = tud.DataLoader(Dataset, num_workers=4, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    epoch_loss = 0
    start_time = time.time()
    for i, (input_meas_H1, input_meas_L, pattern, GT) in enumerate(loader_train):
        input_meas_H1, input_meas_L, pattern, GT = Variable(input_meas_H1), Variable(input_meas_L), Variable(pattern), Variable(GT)
        input_meas_H1, input_meas_L, pattern, GT = input_meas_H1.cuda(), input_meas_L.cuda(), pattern.cuda(), GT.cuda()

        optimizer.zero_grad(set_to_none=True)
        if 'CSST' in opt.method:
            model_out = model(input_meas_H1, input_meas_L, pattern)
            loss = torch.sqrt(mse(model_out, GT))
        elif opt.method in ['cst_s', 'cst_m', 'cst_l']:
            model_out, diff_pred = model(input_meas_H1, input_meas_L, pattern)
            loss = torch.sqrt(mse(model_out, GT))
            diff_gt = torch.mean(torch.abs(model_out.detach() - GT), dim=1, keepdim=True)
            loss_sparsity = F.mse_loss(diff_gt, diff_pred)
            loss = loss + 2 * loss_sparsity
        else:
            model_out = model(input_meas_H1, input_meas_L, pattern)
            loss = torch.sqrt(mse(model_out, GT))
        if opt.method == 'hdnet':
            fdl_loss = FDL_loss(model_out, GT)
            loss = loss + 0.7 * fdl_loss
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    scheduler.step()
    end_time = time.time()
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, epoch_loss / len(Dataset), (end_time - start_time)))
    # torch.save(model, os.path.join(opt.outf, 'model_%03d.pth' % (epoch + 1)))
    return 0

def test(epoch, logger):
    psnr_list, ssim_list = [], []
    test_gt_mea = test_data.cuda().float()
    input_meas_H1, input_meas_L, pattern, test_gt = init_input_adis(test_gt_mea)
    model.eval()
    begin = time.time()
    with torch.no_grad():
        if opt.method in ['cst_s', 'cst_m', 'cst_l']:
            model_out, _ = model(input_meas_H1, input_meas_L, pattern)
        elif 'CSST' in opt.method:
            model_out = model(input_meas_H1, input_meas_L, pattern)
        else:
            model_out = model(input_meas_H1, input_meas_L, pattern)

    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,(end - begin)))
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            print(np.asarray(psnr_all))
            if psnr_mean > 28:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                checkpoint(model, epoch, model_path, logger)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()




