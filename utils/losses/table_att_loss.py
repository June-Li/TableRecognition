from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F


class TableAttentionLoss(nn.Module):
    def __init__(self, structure_weight, loc_weight, use_giou=False, giou_weight=1.0, **kwargs):
        super(TableAttentionLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='none')
        self.structure_weight = structure_weight
        self.loc_weight = loc_weight
        self.use_giou = use_giou
        self.giou_weight = giou_weight
        
    def giou_loss(self, preds, bbox, eps=1e-7, reduction='mean'):
        '''
        :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :return: loss
        '''
        # ix1 = fluid.layers.elementwise_max(preds[:, 0], bbox[:, 0])
        # iy1 = fluid.layers.elementwise_max(preds[:, 1], bbox[:, 1])
        # ix2 = fluid.layers.elementwise_min(preds[:, 2], bbox[:, 2])
        # iy2 = fluid.layers.elementwise_min(preds[:, 3], bbox[:, 3])
        ix1 = (preds[:, 0] >= bbox[:, 0]) * preds[:, 0] + (preds[:, 0] < bbox[:, 0]) * bbox[:, 0]
        iy1 = (preds[:, 1] >= bbox[:, 1]) * preds[:, 1] + (preds[:, 1] < bbox[:, 1]) * bbox[:, 1]
        ix2 = (preds[:, 2] >= bbox[:, 2]) * bbox[:, 2] + (preds[:, 2] < bbox[:, 2]) * preds[:, 2]
        iy2 = (preds[:, 3] >= bbox[:, 3]) * bbox[:, 3] + (preds[:, 3] < bbox[:, 3]) * preds[:, 3]

        iw = torch.clamp(ix2 - ix1 + 1e-3, 0., 1e10)
        ih = torch.clamp(iy2 - iy1 + 1e-3, 0., 1e10)

        # overlap
        inters = iw * ih

        # union
        uni = (preds[:, 2] - preds[:, 0] + 1e-3) * (preds[:, 3] - preds[:, 1] + 1e-3
            ) + (bbox[:, 2] - bbox[:, 0] + 1e-3) * (
            bbox[:, 3] - bbox[:, 1] + 1e-3) - inters + eps

        # ious
        ious = inters / uni

        # ex1 = fluid.layers.elementwise_min(preds[:, 0], bbox[:, 0])
        # ey1 = fluid.layers.elementwise_min(preds[:, 1], bbox[:, 1])
        # ex2 = fluid.layers.elementwise_max(preds[:, 2], bbox[:, 2])
        # ey2 = fluid.layers.elementwise_max(preds[:, 3], bbox[:, 3])
        ex1 = (preds[:, 0] >= bbox[:, 0]) * bbox[:, 0] + (preds[:, 0] < bbox[:, 0]) * preds[:, 0]
        ey1 = (preds[:, 1] >= bbox[:, 1]) * bbox[:, 1] + (preds[:, 1] < bbox[:, 1]) * preds[:, 1]
        ex2 = (preds[:, 2] >= bbox[:, 2]) * preds[:, 2] + (preds[:, 2] < bbox[:, 2]) * bbox[:, 2]
        ey2 = (preds[:, 3] >= bbox[:, 3]) * preds[:, 3] + (preds[:, 3] < bbox[:, 3]) * bbox[:, 3]
        ew = torch.clamp(ex2 - ex1 + 1e-3, 0., 1e10)
        eh = torch.clamp(ey2 - ey1 + 1e-3, 0., 1e10)

        # enclose erea
        enclose = ew * eh + eps
        giou = ious - (enclose - uni) / enclose

        loss = 1 - giou

        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)
        else:
            raise NotImplementedError
        return loss

    def forward(self, predicts, batch):
        structure_probs = predicts['structure_probs']
        # structure_targets = batch[1].astype("int64")
        structure_targets = torch.tensor(batch[1], dtype=torch.int64)
        structure_targets = structure_targets[:, 1:]
        if len(batch) == 6:
            # structure_mask = batch[5].astype("int64")
            structure_mask = torch.tensor(batch[5], dtype=torch.int64)
            structure_mask = structure_mask[:, 1:]
            structure_mask = torch.reshape(structure_mask, [-1])
        structure_probs = torch.reshape(structure_probs, [-1, structure_probs.shape[-1]])
        structure_targets = torch.reshape(structure_targets, [-1])
        structure_loss = self.loss_func(structure_probs, structure_targets)
        
        if len(batch) == 6:
             structure_loss = structure_loss * structure_mask
            
#         structure_loss = paddle.sum(structure_loss) * self.structure_weight
        structure_loss = torch.mean(structure_loss) * self.structure_weight
        
        loc_preds = predicts['loc_preds']
        # loc_targets = batch[2].astype("float32")
        loc_targets = torch.tensor(batch[2], dtype=torch.float32)
        # loc_targets_mask = batch[4].astype("float32")
        loc_targets_mask = torch.tensor(batch[4], dtype=torch.float32)
        loc_targets = loc_targets[:, 1:, :]
        loc_targets_mask = loc_targets_mask[:, 1:, :]
        loc_loss = F.mse_loss(loc_preds * loc_targets_mask, loc_targets) * self.loc_weight
        if self.use_giou:
            loc_loss_giou = self.giou_loss(loc_preds * loc_targets_mask, loc_targets) * self.giou_weight
            total_loss = structure_loss + loc_loss + loc_loss_giou
            return {'loss':total_loss, "structure_loss":structure_loss, "loc_loss":loc_loss, "loc_loss_giou":loc_loss_giou}
        else:
            total_loss = structure_loss + loc_loss            
            return {'loss':total_loss, "structure_loss":structure_loss, "loc_loss":loc_loss}