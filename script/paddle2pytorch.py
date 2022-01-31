import torch
import paddle
import numpy as np
import yaml


config = yaml.load(open('0.yml', 'r'), Loader=yaml.FullLoader)  # 0.yml是配置文件，eg：configs/train_table_mv3.yml
pd = paddle.load('/Volumes/my_disk/company/xxx/buffer_disk/a/best_accuracy.pdparams')
ckpt = torch.load('/Volumes/my_disk/company/xxx/buffer_disk/a/last.pt', map_location='cpu')  # ckpt表示pytorch模型

out_dict = {}
for pd_k in pd.keys():
    ckpt_k = pd_k.replace('stage', 'stages_pipline.').replace('_mean', 'running_mean').replace('_variance', 'running_var')
    if np.shape(pd[pd_k].numpy()) != np.shape(ckpt[ckpt_k].numpy()) or pd_k == 'head.structure_attention_cell.h2h.weight':
        pd[pd_k] = paddle.transpose(pd[pd_k], (1, 0))
    out_dict[ckpt_k] = torch.tensor(pd[pd_k].numpy())
torch.save({'state_dict': out_dict, 'cfg': config}, '/Volumes/my_disk/company/xxx/buffer_disk/a/pd2pt.pt')