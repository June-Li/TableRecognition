# -*- coding: utf-8 -*-
# @Time    : 2022/01/25 09:52
# @Author  : lijun

import torch


# def load_model(model, model_path):
#     model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
#
#
# def save_model(save_dict, save_path):
#     torch.save(save_dict, save_path)


def load_model(model, model_path):
    state_dict = {}
    for k, v in torch.load(model_path, map_location='cpu')['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    model.load_state_dict(state_dict)


def save_model(save_dict, save_path):
    state_dict = {}
    for k, v in save_dict['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    save_dict['state_dict'] = state_dict
    torch.save(save_dict, save_path)
