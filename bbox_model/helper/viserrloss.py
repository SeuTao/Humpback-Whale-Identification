import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class VisErrorLoss(nn.Module):
    def __init__(self):
        super(VisErrorLoss, self).__init__()

    def compute_l1_weighted_loss(self, hm_targets, hm_preds, vismap, ohem=1.0):
        '''
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number]
        :return: 
        '''
        epsilon = 0.0001
        hm_preds = F.relu(hm_preds, False)
        amplitude = torch.max(hm_targets)
        b, k, h, w = hm_targets.size()
        hm_targets = hm_targets.view(b, k, -1)
        hm_preds = hm_preds.view(b, k, -1)
        vismap = vismap.view(b, k, 1).repeat(1, 1, h * w)
        pos_ids = ((hm_targets > (amplitude / 10)) & (vismap >= 0))  # variable with requires_grad=False
        neg_ids = ((hm_targets <= (amplitude / 10)) & (vismap >= 0))
        diff = (hm_targets - hm_preds).abs()
        pos_loss = (diff * pos_ids.float()).sum(2).sum(0) / (pos_ids.float().sum(2).sum(0) + epsilon)
        neg_loss = (diff * neg_ids.float()).sum(2).sum(0) / (neg_ids.float().sum(2).sum(0) + epsilon)
        total_loss = 0.5 * pos_loss + 0.5 * neg_loss
        if ohem < 1:
            k = int(total_loss.size(0) * ohem)
            total_loss, _ = total_loss.topk(k)
        return total_loss.mean()

    def compute_l2_loss(self, hm_targets, hm_preds, vismap, ohem=1.0):
        '''
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number]
        :return: 
        '''
        epsilon = 0.0001
        hm_preds = F.relu(hm_preds, False)
        b, k, h, w = hm_targets.size()
        hm_targets = hm_targets.view(b, k, -1)
        hm_preds = hm_preds.view(b, k, -1)
        vismap = vismap.view(b, k, 1).repeat(1, 1, h * w)
        ids = vismap == 1  # variable with requires_grad=False
        diff = (hm_targets - hm_preds)**2
        total_loss = (diff * ids.float()).sum(2).sum(0) / (ids.float().sum(2).sum(0) + epsilon)
        if ohem < 1:
            k = int(total_loss.size(0) * ohem)
            total_loss, _ = total_loss.topk(k)
        return total_loss.mean()

    def forward(self, hm_targets, hm_preds1, hm_preds2, vismap):
        '''
        :param hm_targets: [batch size, keypoint number, h, w]
        :param hm_preds: [batch size, keypoint number, h, w]
        :param vismap: [batch size, keypoint number] 
        :return: 
        '''
        loss1 = self.compute_l1_weighted_loss(hm_targets, hm_preds1, vismap)
        loss2 = self.compute_l1_weighted_loss(hm_targets, hm_preds2, vismap, ohem=0.5)
        return loss1+loss2, loss1, loss2

