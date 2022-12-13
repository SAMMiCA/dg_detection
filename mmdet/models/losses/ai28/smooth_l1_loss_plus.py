# Copyright (c) OpenMMLab. All rights reserved.

# Modifications:
# Copyright (c) 2022 Urban Robotics Lab. @ KAIST. All rights reserved.
import mmcv
import torch
import torch.nn as nn

from ...builder import LOSSES
from ..utils import weighted_loss, weighted_loss2
import torch.nn.functional as F
from ..utils import weight_reduce_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss2
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    pred_orig, _, _ = torch.chunk(pred, 3)
    target, _, _ = torch.chunk(target, 3)

    assert beta > 0
    if target.numel() == 0:
        return pred_orig.sum() * 0

    assert pred_orig.size() == target.size()
    diff = torch.abs(pred_orig - target)
    loss_orig = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)

    return loss_orig


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss2
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    pred_orig, _, _ = torch.chunk(pred, 3)
    target, _, _ = torch.chunk(target, 3)

    if target.numel() == 0:
        return pred_orig.sum() * 0

    assert pred_orig.size() == target.size()
    loss_orig = torch.abs(pred_orig - target)

    return loss_orig


def smooth_l1_dg_loss(pred, target, beta=1.0, weight=None, reduction=None, avg_factor=None):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    target, _, _ = torch.chunk(target, 3)

    assert beta > 0
    if target.numel() == 0:
        return pred_orig.sum() * 0

    assert pred_orig.size() == pred_aug1.size()
    diff = (torch.abs(pred_orig - pred_aug1) + torch.abs(pred_orig - pred_aug2)) / 2
    additional_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    additional_loss = additional_loss.mean()

    p_distribution = {'pred_orig': pred_orig,
                      'pred_aug1': pred_aug1,
                      'pred_aug2': pred_aug2,
                      }

    return additional_loss, p_distribution


def l1_dg_loss(pred, target, weight=None, reduction=None, avg_factor=None):
    """L1 dg loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    target, _, _ = torch.chunk(target, 3)

    if target.numel() == 0:
        return pred_orig.sum() * 0

    assert pred_orig.size() == pred_aug1.size()
    additional_loss = (torch.abs(pred_orig - pred_aug1) + torch.abs(pred_aug1 - pred_aug2) + torch.abs(pred_aug2 - pred_orig)) / 3
    additional_loss = additional_loss.mean()

    p_distribution = {'pred_orig': pred_orig,
                      'pred_aug1': pred_aug1,
                      'pred_aug2': pred_aug2,
                      }

    return additional_loss, p_distribution


def jsd(pred,
        label,
        weight=None,
        reduction='mean',
        avg_factor=None):
    """Calculate the jsd loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    # label, _, _ = torch.chunk(label, 3)

    p_clean, p_aug1, p_aug2 = F.softmax(
        pred_orig, dim=1), F.softmax(
        pred_aug1, dim=1), F.softmax(
        pred_aug2, dim=1)
    p_clean, p_aug1, p_aug2 = p_clean.reshape((1,) + p_clean.shape).contiguous(), \
                              p_aug1.reshape((1,) + p_aug1.shape).contiguous(), \
                              p_aug2.reshape((1,) + p_aug2.shape).contiguous()

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    # apply weights and do the reduction
    if weight is not None:
        weight, _, _ = torch.chunk(weight, 3)
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    p_distribution = {'p_clean':p_clean,
                      'p_aug1':p_aug1,
                      'p_aug2':p_aug2,
                      'p_mixture':p_mixture}

    return loss, p_distribution



def jsdy(pred,
         label,
         weight=None,
         reduction='mean',
         avg_factor=None):
    """Calculate the jsdy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    label, _, _ = torch.chunk(label, 3)

    if pred_orig.shape != label.shape:
        if pred_orig.shape[-1] is 1: # if rpn
            label = label.reshape(label.shape+(1,)).contiguous()
        else: # else roi
            label = F.one_hot(label, num_classes=pred_orig.shape[-1]) # TO-DO: need to check

    p_clean, p_aug1, p_aug2 = F.softmax(
        pred_orig, dim=1), F.softmax(
        pred_aug1, dim=1), F.softmax(
        pred_aug2, dim=1)
    p_clean, p_aug1, p_aug2 = p_clean.reshape((1,) + p_clean.shape).contiguous(), \
                              p_aug1.reshape((1,) + p_aug1.shape).contiguous(), \
                              p_aug2.reshape((1,) + p_aug2.shape).contiguous()
    label = label.reshape((1,) + label.shape).contiguous()
    label = label.type(torch.cuda.FloatTensor)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2 + label.contiguous()) / 4., 1e-7, 1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug2, reduction='batchmean') +
            F.kl_div(p_mixture, label, reduction='batchmean')) / 4.

    # apply weights and do the reduction
    if weight is not None:
        weight, _, _ = torch.chunk(weight, 3)
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    p_distribution = {'p_clean': p_clean,
                      'p_aug1': p_aug1,
                      'p_aug2': p_aug2,
                      'p_mixture': p_mixture}

    return loss, p_distribution


@LOSSES.register_module()
class SmoothL1LossPlus(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0,
                 additional_loss='jsd',
                 lambda_weight=0.0001,
                 wandb_name=None
                 ):
        super(SmoothL1LossPlus, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.additional_loss = additional_loss
        self.lambda_weight = lambda_weight
        self.wandb_name = wandb_name

        self.wandb_features = dict()
        self.wandb_features[f'additional_loss({self.wandb_name})'] = []
        self.wandb_features[f'smoothL1_loss({self.wandb_name})'] = []

        if self.additional_loss == 'jsd':
            self.cls_additional = jsd
        elif self.additional_loss == 'jsdy':
            self.cls_additional = jsdy
        elif self.additional_loss == 'l1_dg_loss':
            self.cls_additional = l1_dg_loss
        elif self.additional_loss == 'smooth_l1_dg_loss':
            self.cls_additional = smooth_l1_dg_loss
        else:
            self.cls_additional = None

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        loss_additional = 0
        if self.cls_additional is not None:
            loss_additional, p_distribution = self.cls_additional(
                pred,
                target,
                weight,
                reduction=reduction,
                avg_factor=avg_factor)

            self.wandb_features[f'smoothL1_loss({self.wandb_name})'] = loss_bbox
            self.wandb_features[f'additional_loss({self.wandb_name})'] = loss_additional
            for key, value in p_distribution.items():
                self.wandb_features[f'{key}({self.wandb_name})'] = value

        loss = loss_bbox + self.lambda_weight * loss_additional
        self.wandb_features[f'loss({self.wandb_name})'] = loss
        return loss


@LOSSES.register_module()
class L1LossPlus(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0,
                 additional_loss='jsd',
                 lambda_weight=0.0001,
                 wandb_name=None
                 ):
        super(L1LossPlus, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.additional_loss = additional_loss
        self.lambda_weight = lambda_weight
        self.wandb_name = wandb_name

        self.wandb_features = dict()
        self.wandb_features[f'additional_loss({self.wandb_name})'] = []
        self.wandb_features[f'L1_loss({self.wandb_name})'] = []

        if self.additional_loss == 'jsd':
            self.cls_additional = jsd
        elif self.additional_loss == 'jsdy':
            self.cls_additional = jsdy
        elif self.additional_loss == 'l1_dg_loss':
            self.cls_additional = l1_dg_loss
        elif self.additional_loss == 'smooth_l1_dg_loss':
            self.cls_additional = smooth_l1_dg_loss
        else:
            self.cls_additional = None

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)

        loss_additional = 0
        if self.cls_additional is not None:
            loss_additional, p_distribution = self.cls_additional(
                pred,
                target,
                weight,
                reduction=reduction,
                avg_factor=avg_factor)

            # wandb for rpn
            if len(self.wandb_features[f'L1_loss({self.wandb_name})']) == 5:
                self.wandb_features[f'L1_loss({self.wandb_name})'].clear()
                self.wandb_features[f'additional_loss({self.wandb_name})'].clear()
            self.wandb_features[f'L1_loss({self.wandb_name})'].append(loss_bbox)
            self.wandb_features[f'additional_loss({self.wandb_name})'].append(
                self.lambda_weight * loss_additional)

            for key, value in p_distribution.items():
                self.wandb_features[f'{key}({self.wandb_name})'] = value

        loss = loss_bbox + self.lambda_weight * loss_additional
        self.wandb_features[f'loss({self.wandb_name})'] = loss
        return loss
