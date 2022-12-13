# Copyright (c) OpenMMLab. All rights reserved.

# Modifications:
# Copyright (c) 2022 Urban Robotics Lab. @ KAIST. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import LOSSES
from ..utils import weight_reduce_loss


def fpn_loss(pred,
             weight=None,
             reduction='mean',
             avg_factor=None,
             ignore_index=-100,
             **kwargs):
    """Calculate the jsdfp loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        temper (int, optional): temperature scaling for softmax function.

    Returns:
        torch.Tensor: The calculated loss
    """
    p_clean, p_aug1, p_aug2 = 0, 0, 0
    temper = kwargs['temper'] if 'temper' in kwargs else 1
    add_act = kwargs['add_act'] if 'add_act' in kwargs else None
    loss_type = kwargs['loss_type'] if 'loss_type' in kwargs else 'mse'

    # chunk the data to get p_orig
    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    pred_orig, pred_aug1, pred_aug2 = pred_orig.view(-1, 1), pred_aug1.view(-1, 1), pred_aug2.view(-1, 1)

    if add_act == None:
        # sigmoid and softmax function for rpn_cls and roi_cls
        if pred_orig.shape[-1] is 1:    # if rpn
            p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig / temper), \
                                      torch.sigmoid(pred_aug1 / temper), \
                                      torch.sigmoid(pred_aug2 / temper)
        else:   # else roi
            p_clean, p_aug1, p_aug2 = F.softmax(pred_orig / temper, dim=1), \
                                      F.softmax(pred_aug1 / temper, dim=1), \
                                      F.softmax(pred_aug2 / temper, dim=1)
    elif add_act == 'softmax':
        p_clean, p_aug1, p_aug2 = F.softmax(pred_orig / temper, dim=1), \
                                  F.softmax(pred_aug1 / temper, dim=1), \
                                  F.softmax(pred_aug2 / temper, dim=1)
    elif add_act == 'sigmoid':
        p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig / temper), \
                                  torch.sigmoid(pred_aug1 / temper), \
                                  torch.sigmoid(pred_aug2 / temper)
    elif add_act == 'pred':
        p_clean, p_aug1, p_aug2 = pred_orig, pred_aug1, pred_aug2

    if loss_type == 'jsd':
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        loss = (F.kl_div(p_mixture, p_clean, reduction='none') +
                F.kl_div(p_mixture, p_aug1, reduction='none') +
                F.kl_div(p_mixture, p_aug2, reduction='none')) / 3.

    elif loss_type == 'mse':
        loss = (F.mse_loss(p_clean, p_aug1, reduction='none'),
                F.mse_loss(p_clean, p_aug2, reduction='none'),
                F.mse_loss(p_aug1, p_aug2, reduction='none'))

    # apply weights and do the reduction
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=None)  # avg_factor=avg_factor is deprecated

    if weight is not None:
        assert p_clean.size() == weight.size(), \
            "The size of tensors does not match"
        # get valid predictions for wandb log
        p_clean, p_aug1, p_aug2= torch.clamp(p_clean[weight!=0], 1e-7, 1).log(), \
                                             torch.clamp(p_aug1[weight!=0], 1e-7, 1).log(), \
                                             torch.clamp(p_aug2[weight!=0], 1e-7, 1).log(),

    # logging
    p_distribution = {'p_clean': p_clean,
                      'p_aug1': p_aug1,
                      'p_aug2': p_aug2,
                      }

    return loss, p_distribution
