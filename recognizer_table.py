# -*- coding: utf-8 -*-
# @Time    : 2022/01/25 09:52
# @Author  : lijun
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import platform
import time
import cv2
import torch
import numpy as np
from tqdm import tqdm

from models.architectures import build_model
from utils.save_load import load_model
from utils.torch_utils import select_device
from utils.data import transform, create_operators
from utils.postprocess import build_post_process


class Recognizer:
    def __init__(self, model_path, table_char_dict_path='utils/dict/table_structure_dict.txt',
                 gpu='0', half_flag=False):
        self.weights = model_path
        self.table_char_dict_path = table_char_dict_path
        self.gpu = gpu
        self.device = select_device(self.gpu)
        self.half_flag = half_flag
        ckpt = torch.load(self.weights, map_location='cpu')
        cfg = ckpt['cfg']

        pre_process_list = []
        for pre_process in cfg['Train']['dataset']['transforms']:
            if 'DecodeImage' in pre_process.keys() or 'TableLabelEncode' in pre_process.keys():
                continue
            if 'KeepKeys' in pre_process.keys():
                pre_process['KeepKeys'] = {'keep_keys': ['image']}
            pre_process_list.append(pre_process)
        self.preprocess_op = create_operators(pre_process_list)
        postprocess_params = {
            'name': 'TableLabelDecode',
            "character_type": 'en',
            "character_dict_path": self.table_char_dict_path,
        }
        self.postprocess_op = build_post_process(postprocess_params)

        self.model = build_model(cfg['Architecture'])
        load_model(self.model, model_path)
        self.model.to(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half_flag and self.half:
            self.model.half()  # to FP16

    def inference(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]
        if img is None:
            return None
        img = np.expand_dims(img, axis=0)
        img = img.copy()
        img = torch.tensor(img).to(self.device)
        img = img.half() if self.half_flag and self.half else img
        preds = self.model(img)

        preds = {'structure_probs': preds['structure_probs'].cpu().detach().numpy(),
                 'loc_preds': preds['loc_preds'].cpu().detach().numpy()}
        post_result = self.postprocess_op(preds)

        structure_str_list = post_result['structure_str_list']
        res_loc = post_result['res_loc']
        imgh, imgw = ori_im.shape[0:2]
        res_loc_final = []
        for rno in range(len(res_loc[0])):
            x0, y0, x1, y1 = res_loc[0][rno]
            left = max(int(imgw * x0), 0)
            top = max(int(imgh * y0), 0)
            right = min(int(imgw * x1), imgw - 1)
            bottom = min(int(imgh * y1), imgh - 1)
            res_loc_final.append([left, top, right, bottom])
        structure_str_list = structure_str_list[0][:-1]
        structure_str_list = ['<html>', '<body>', '<table>'] + structure_str_list + ['</table>', '</body>', '</html>']
        return structure_str_list, res_loc_final


if __name__ == '__main__':
    def html_configuration():
        html_head = '<!DOCTYPE html>\n' + \
                    '<html lang="en">\n' + \
                    '<head>\n' + \
                    '    <meta charset="UTF-8">\n' + \
                    '    <title>Title</title>\n' + \
                    '</head>\n' + \
                    '<body>\n' + \
                    '<style type="text/css">\n' + \
                    '.tg  {border-collapse:collapse;border-spacing:0;}\n' + \
                    '.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;\n' + \
                    '  overflow:hidden;padding:10px 5px;word-break:normal;}\n' + \
                    '.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;\n' + \
                    '  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}\n' + \
                    '.tg .tg-0lax{text-align:left;vertical-align:top}\n' + \
                    '</style>\n' + \
                    '<table class="tg">\n'
        html_tail = '</table>\n' + \
                    '</body>\n' + \
                    '</html>\n'
        return [html_head, html_tail]


    # table_rec = Recognizer('inference_model/baidu/infer/table_rec/pd2pt.pt', gpu='1')
    table_rec = Recognizer('inference_model/juneli/finetune/table_rec/last.pt', gpu='cpu')
    base_path = './'
    in_path = base_path + 'test_data/FundScan/'
    out_path = base_path + 'test_out/FundScan/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # file_name_list = os.listdir(base_path + 'datasets/public/pubtabnet_subset/train/')
    file_name_list = os.listdir(in_path)
    for idx, file_name in enumerate(tqdm(file_name_list)):
        # if not file_name.startswith('0.jpg'):
        #     continue
        # if file_name != '0.jpg':
        #     continue
        img = cv2.imread(in_path + file_name)
        structure, loc = table_rec.inference(img)
        # my output->开始
        # print('html: \n', structure)
        open(out_path + file_name.replace('.jpg', '.html').replace('.png', '.html'), 'w').write(
            html_configuration()[0] +
            ''.join([i.replace('"', '"') for i in structure[3:-3]]) + '\n' + html_configuration()[1])
        print(loc)
        # print(len(structure), len(loc))
        show_img = img.copy()
        for box in loc:
            cv2.rectangle(show_img, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), 2)
        cv2.imwrite(out_path + file_name, show_img)
        # my output->结束


