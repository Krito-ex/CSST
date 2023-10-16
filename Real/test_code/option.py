import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='CSST_5stg',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')
parser.add_argument('--data_path_CAVE', default='../../datasets/CAVE_1024_28_30/', type=str,
                    help='path of data')
parser.add_argument('--data_path_KAIST', default='../../datasets/KAIST_CVPR2021/', type=str,
                    help='path of data')


# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/CSST_stg/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='CSST_5stg', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')

# Training specifications
parser.add_argument('--batch_size', type=int, default=4, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=1250, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)
# parser.add_argument("--local_rank", type=int)

opt = parser.parse_args()   #args
template.set_template(opt)

# dataset
# opt.data_path = f"{opt.data_root}/cave_1024_28/"

opt.data_path = f"{opt.data_root}CAVE_1024_28_30/"
# opt.mask_path = f"{opt.data_root}/TSA_simu_data/"
opt.mask_path = f"{opt.data_root}ADIS/"
opt.test_path = f"{opt.data_root}/KAIST_926/"
# opt.test_path = f"{opt.data_root}CAVE_512_28_test/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False






