import torch
from .CSST import CSST

def model_generator(method, pretrained_model_path=None):
    if 'CSST' in method:
        num_iterations = int(method.split('_')[1][0]) 
        model = CSST(num_iterations=num_iterations).cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    model = torch.nn.DataParallel(model)
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k: v for k, v in checkpoint.items()},
                              strict=True)
    return model
