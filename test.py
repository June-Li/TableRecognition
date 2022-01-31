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
import torch
from tqdm import tqdm

from models.architectures import build_model
from utils.save_load import load_model
from utils.metrics import build_metric
from utils.data import build_dataloader
from utils.utility import preprocess


def test():
    """
    :param: None
    :return: 准确率
    """

    """ 读取配置文件--->开始 """
    config, device, logger = preprocess(is_train=True)
    """ 读取配置文件--->结束 """

    """ 构建dataloader--->开始 """
    valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    if len(valid_dataloader) == 0:
        logger.error("please check val_dataloader\n")
        return
    """ 构建dataloader--->结束 """

    """ 构建模型--->开始 """
    model = build_model(config['Architecture'])
    load_model(model, config['Global']['checkpoints'])
    model.to(device)
    """ 构建模型--->结束 """

    """ 构建metric--->开始 """
    eval_class = build_metric(config['Metric'])
    """ 构建metric--->结束 """

    """ test--->开始 """
    model.eval()
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(total=len(valid_dataloader), desc='eval model:')
        max_iter = len(valid_dataloader) - 1 if platform.system() == "Windows" else len(valid_dataloader)
        for idx, batch in enumerate(valid_dataloader):
            batch = [i.to(device) for i in batch]
            if idx >= max_iter:
                break
            images = batch[0]
            start = time.time()

            preds = model(images, data=batch[1:])

            # Obtain usable results from post-processing methods
            total_time += time.time() - start
            # Evaluate the results of the current batch

            preds = {'structure_probs': preds['structure_probs'].cpu().detach().numpy(),
                     'loc_preds': preds['loc_preds'].cpu().detach().numpy()}
            batch = [item.cpu().numpy() for item in batch]
            eval_class(preds, batch)

            pbar.update(1)
            total_frame += len(images)
        # Get final metric->acc
        metric = eval_class.get_metric()
    """ test--->结束 """

    pbar.close()
    model.train()
    metric['fps'] = total_frame / total_time
    return metric


if __name__ == '__main__':
    metric = test()
    print('metric: ', metric)
