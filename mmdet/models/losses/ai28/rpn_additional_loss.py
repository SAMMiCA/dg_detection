# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import LOSSES
from ..utils import weight_reduce_loss


def jsd(features,
        **kwargs):
    features = features.reshape(features.shape[0], -1)
    assert features.shape[0] == 3

    feats_orig, feats_aug1, feats_aug2 = torch.chunk(features, 3)
    p_clean, p_aug1, p_aug2 = F.softmax(feats_orig, dim=1), \
                              F.softmax(feats_aug1, dim=1), \
                              F.softmax(feats_aug2, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    kld1 = F.kl_div(p_mixture, p_clean, reduction='batchmean')
    kld2 = F.kl_div(p_mixture, p_aug1, reduction='batchmean')
    kld3 = F.kl_div(p_mixture, p_aug2, reduction='batchmean')

    loss = (kld1 + kld2 + kld3) / 3.

    return loss


@LOSSES.register_module()
class RpnAdditionalLoss(nn.Module):

    def __init__(self,
                 loss_additional,
                 **kwargs):
        super(RpnAdditionalLoss, self).__init__()

        self.kwargs = kwargs
        self.version = loss_additional['version']
        self.weight = loss_additional['weight'] if 'weight' in loss_additional else 12
        self.wandb_name = loss_additional['wandb_name'] if 'wandb_name' in loss_additional else 'rpn_additional'
        self.wandb_features = dict()
        self.wandb_features[self.wandb_name] = []

    def forward(self,
                rpn_additional,
                **kwargs):
        losses = []
        if self.version == '2.1' or '2.2':
            criterion = jsd
        else:
            raise ValueError(f"version must be ['2.1'],"
                             f"but got {self.version}.")

        for i in range(len(rpn_additional)):
            loss = criterion(rpn_additional[i])
            losses.append(self.weight * loss)

            if len(self.wandb_features[self.wandb_name]) == 5:
                self.wandb_features[self.wandb_name].clear()
            self.wandb_features[self.wandb_name].append(self.weight * loss)
        return losses

