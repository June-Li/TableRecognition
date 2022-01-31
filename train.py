# -*- coding: utf-8 -*-
# @Time    : 2022/01/25 09:52
# @Author  : lijun

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import platform
import yaml
import time
import shutil
import torch
from tqdm import tqdm
import numpy as np

from models.architectures import build_model
from utils.save_load import load_model
from utils.losses import build_loss
from utils.optimizer import build_optimizer
from utils.metrics import build_metric
from utils.stats import TrainingStats
from utils.save_load import save_model
from utils.data import build_dataloader
from utils.utility import preprocess


def eval(model,
         valid_dataloader,
         eval_class,
         device):
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

    pbar.close()
    model.train()
    metric['fps'] = total_frame / total_time
    return metric


def train():
    """
    :param：None
    :return: None
    """

    """ 读取配置文件--->开始 """
    config, device, logger = preprocess(is_train=True)
    """ 读取配置文件--->结束 """

    """ 构建dataloader--->开始 """
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    if len(train_dataloader) == 0:
        logger.error("please check train_dataloader\n")
        return
    if len(valid_dataloader) == 0:
        logger.error("please check val_dataloader\n")
        return
    """ 构建dataloader--->结束 """

    """ 构建模型--->开始 """
    model = build_model(config['Architecture'])
    if config['Global']['checkpoints']:
        load_model(model, config['Global']['checkpoints'])
    model.to(device)
    model = torch.nn.DataParallel(model)
    """ 构建模型--->结束 """

    """ 构建loss--->开始 """
    loss_class = build_loss(config['Loss'])
    """ 构建loss--->结束 """

    """ 构建optim--->开始 """
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        parameters=model.parameters())
    """ 构建optim--->结束 """

    """ 构建metric--->开始 """
    eval_class = build_metric(config['Metric'])
    """ 构建metric--->结束 """

    """ 其他---> """
    cal_metric_during_train = config['Global'].get('cal_metric_during_train', False)
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']

    global_step = 0
    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    train_stats = TrainingStats(log_smooth_window, ['lr'])
    """ 其他---> """

    """ trian--->开始 """
    model.train()
    for epoch in range(1, epoch_num + 1):
        train_dataloader = build_dataloader(config, 'Train', device, logger, seed=epoch)
        train_batch_cost = 0.0
        train_reader_cost = 0.0
        batch_sum = 0
        batch_start = time.time()
        max_iter = len(train_dataloader) - 1 if platform.system() == "Windows" else len(train_dataloader)
        for idx, batch in enumerate(train_dataloader):
            batch = [i.to(device) for i in batch]
            train_reader_cost += time.time() - batch_start
            if idx >= max_iter:
                break
            lr = optimizer.defaults['lr']
            images = batch[0]

            preds = model(images, data=batch[1:])

            loss = loss_class(preds, batch)
            avg_loss = loss['loss']
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['Optimizer']['clip_norm'])
            optimizer.step()
            optimizer.zero_grad()

            train_batch_cost += time.time() - batch_start
            batch_sum += len(images)

            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

            stats = {k: v.cpu().detach().numpy().mean() for k, v in loss.items()}
            stats['lr'] = lr
            train_stats.update(stats)

            if cal_metric_during_train:  # only rec and cls need
                batch = [item.cpu().numpy() for item in batch]
                preds = {'structure_probs': preds['structure_probs'].cpu().detach().numpy(),
                         'loc_preds': preds['loc_preds'].cpu().detach().numpy()}
                eval_class(preds, batch)
                metric = eval_class.get_metric()
                train_stats.update(metric)

            if (global_step > 0 and global_step % print_batch_step == 0) or (idx >= len(train_dataloader) - 1):
                logs = train_stats.log()
                strs = 'epoch: [{}/{}], iter: {}, {}, reader_cost: {:.5f} s, batch_cost: {:.5f} s, samples: {}, ips: {:.5f}'.format(
                    epoch, epoch_num, global_step, logs, train_reader_cost /
                    print_batch_step, train_batch_cost / print_batch_step,
                    batch_sum, batch_sum / train_batch_cost)
                logger.info(strs)
                train_batch_cost = 0.0
                train_reader_cost = 0.0
                batch_sum = 0
            if global_step % 400 == 0:
                save_model({'state_dict': model.state_dict(), 'cfg': dict(config)},
                           os.path.join(save_model_dir, 'last.pt'))
            # eval
            if global_step % eval_batch_step == 0 and global_step > 0:
                cur_metric = eval(
                    model,
                    valid_dataloader,
                    eval_class,
                    device
                )
                cur_metric_str = \
                    'cur metric, {}'.format(', '.join(['{}: {}'.format(k, v) for k, v in cur_metric.items()]))
                logger.info(cur_metric_str)

                if cur_metric[main_indicator] >= best_model_dict[main_indicator]:
                    best_model_dict.update(cur_metric)
                    best_model_dict['best_epoch'] = epoch
                    save_model({'state_dict': model.state_dict(), 'cfg': dict(config)},
                               os.path.join(save_model_dir, 'best.pt'))
                best_str = \
                    'best metric, {}'.format(', '.join(['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
                logger.info(best_str)

            global_step += 1
            optimizer.zero_grad()
            batch_start = time.time()
        if epoch % save_epoch_step == 0:
            save_model({'state_dict': model.state_dict(), 'cfg': dict(config)},
                       os.path.join(save_model_dir, str(epoch) + '.pt'))
        # if dist.get_rank() == 0:
        #     save_model(
        #         model,
        #         optimizer,
        #         save_model_dir,
        #         logger,
        #         is_best=False,
        #         prefix='latest',
        #         best_model_dict=best_model_dict,
        #         epoch=epoch,
        #         global_step=global_step)
        # if dist.get_rank() == 0 and epoch > 0 and epoch % save_epoch_step == 0:
        #     save_model(
        #         model,
        #         optimizer,
        #         save_model_dir,
        #         logger,
        #         is_best=False,
        #         prefix='iter_epoch_{}'.format(epoch),
        #         best_model_dict=best_model_dict,
        #         epoch=epoch,
        #         global_step=global_step)
    best_str = 'best metric, {}'.format(', '.join(['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
    logger.info(best_str)
    """ trian--->结束 """


if __name__ == '__main__':
    train()
