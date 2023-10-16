
import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='CSST_9stg', help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0, 1')

# Data specifications
# parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')
parser.add_argument('--data_root', type=str, default='/home/root/data1/lvtao/CSST/datasets/', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/CSST_9stg/', help='saving_path')
# parser.add_argument('--outf', type=str, default='./exp/unet/', help='saving_path')

# Model specifications
# parser.add_argument('--method', type=str, default='unet', help='method name')
parser.add_argument('--method', type=str, default='CSST_9stg', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')

# Training specifications
parser.add_argument('--batch_size', type=int, default=4, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)
parser.add_argument("--isTrain", default=True, type=bool, help='train or test')
parser.add_argument("--size", default=256, type=int, help='cropped patch size')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')

opt = parser.parse_args()   #args
template.set_template(opt)
# opt.trainset_num = 20000 // ((opt.size // 64) ** 2)    #1250
opt.trainset_num = 20000 // ((opt.size // 128) ** 2)     #5000
# opt.trainset_num = 1
# dataset

opt.data_path1 = f"{opt.data_root}cave_1024_28/"
# opt.data_path2 = f"{opt.data_root}cave_1024_28_1/"
opt.data_path2 = f"{opt.data_root}KAIST_non_selected/"
opt.test_path = f"{opt.data_root}KAIST_selected_1.3_resize/"


for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False






