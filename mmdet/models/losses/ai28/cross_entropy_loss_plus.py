# Copyright (c) OpenMMLab. All rights reserved.

# Modifications:
# Copyright (c) 2022 Urban Robotics Lab. @ KAIST. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import LOSSES
from ..utils import weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index

    pred_orig, _, _ = torch.chunk(pred, 3)
    label, _, _ = torch.chunk(label, 3)

    loss = F.cross_entropy(
        pred_orig,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    weight, _, _ = torch.chunk(weight, 3)
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0),
                                               label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.

    Returns:
        torch.Tensor: The calculated loss.
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1),
                                              ignore_index)

    # weighted element-wise losses
    weight, _, _ = torch.chunk(weight, 3)
    if weight is not None:
        weight = weight.float()

    pred_orig, _, _ = torch.chunk(pred, 3)
    label, _, _ = torch.chunk(label, 3)

    loss = F.binary_cross_entropy_with_logits(
        pred_orig, label.float(), pos_weight=class_weight, reduction='none')

    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, *), C is the
            number of classes. The trailing * indicates arbitrary shape.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss

    Example:
        >>> N, C = 3, 11
        >>> H, W = 2, 2
        >>> pred = torch.randn(N, C, H, W) * 1000
        >>> target = torch.rand(N, H, W)
        >>> label = torch.randint(0, C, size=(N,))
        >>> reduction = 'mean'
        >>> avg_factor = None
        >>> class_weights = None
        >>> loss = mask_cross_entropy(pred, target, label, reduction,
        >>>                           avg_factor, class_weights)
        >>> assert loss.shape == (1,)
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)

    pred_orig, _, _ = torch.chunk(pred_slice, 3)
    label, _, _ = torch.chunk(label, 3)

    loss = F.binary_cross_entropy_with_logits(
        pred_orig, target, weight=class_weight, reduction='mean')[None]

    return loss


def jsd(pred,
        label,
        weight=None,
        reduction='mean',
        avg_factor=None,
        **kwargs):
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

    p_distribution = {'p_clean': p_clean,
                      'p_aug1': p_aug1,
                      'p_aug2': p_aug2,
                      'p_mixture': p_mixture}

    return loss, p_distribution


def jsdv1_1(pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None,
            **kwargs):
    """Calculate the jsdv1.1 loss.

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

    temper = kwargs['temper']
    add_act = kwargs['add_act']

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)

    if pred_orig.shape[-1] == 1:  # if rpn
        p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig), \
                                  torch.sigmoid(pred_aug1), \
                                  torch.sigmoid(pred_aug2)
    else:  # else roi
        p_clean, p_aug1, p_aug2 = F.softmax(pred_orig, dim=1), \
                                  F.softmax(pred_aug1, dim=1), \
                                  F.softmax(pred_aug2, dim=1)

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

    p_distribution = {'p_clean': torch.clamp(p_clean, 1e-7, 1).log(),
                      'p_aug1': torch.clamp(p_aug1, 1e-7, 1).log(),
                      'p_aug2': torch.clamp(p_aug2, 1e-7, 1).log(),
                      'p_mixture': p_mixture}

    return loss, p_distribution


def jsdv1_2(pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None,
            **kwargs):
    """Calculate the jsdv1.1 loss.

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

    avg_factor = None
    temper = kwargs['temper']
    add_act = kwargs['add_act']

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)

    if pred_orig.shape[-1] == 1:  # if rpn
        p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig), \
                                  torch.sigmoid(pred_aug1), \
                                  torch.sigmoid(pred_aug2)
    else:  # else roi
        p_clean, p_aug1, p_aug2 = F.softmax(pred_orig, dim=1), \
                                  F.softmax(pred_aug1, dim=1), \
                                  F.softmax(pred_aug2, dim=1)

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

    p_distribution = {'p_clean': torch.clamp(p_clean, 1e-7, 1).log(),
                      'p_aug1': torch.clamp(p_aug1, 1e-7, 1).log(),
                      'p_aug2': torch.clamp(p_aug2, 1e-7, 1).log(),
                      'p_mixture': p_mixture}

    return loss, p_distribution


def jsdv1_3(pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None,
            **kwargs):
    """Calculate the jsdv1.3 loss.
    jsd loss (sigmoid, 1-sigmoid) for rpn head, softmax for roi head
    divided by batchmean, divided by 768 (256*3) for rpn, 1056 (352*3) for roi
    reduction parameter does not affect the loss

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

    # avg_factor = None
    temper = kwargs['temper']
    add_act = kwargs['add_act']

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)

    if pred_orig.shape[-1] == 1:  # if rpn
        # p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig), \
        #                           torch.sigmoid(pred_aug1),\
        #                           torch.sigmoid(pred_aug2)
        p_clean, p_aug1, p_aug2 = torch.cat((torch.sigmoid(pred_orig), 1 - torch.sigmoid(pred_orig)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug1), 1 - torch.sigmoid(pred_aug1)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug2), 1 - torch.sigmoid(pred_aug2)), dim=1),

    else:  # else roi
        p_clean, p_aug1, p_aug2 = F.softmax(pred_orig, dim=1), \
                                  F.softmax(pred_aug1, dim=1), \
                                  F.softmax(pred_aug2, dim=1)

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

    p_distribution = {'p_clean': torch.clamp(p_clean, 1e-7, 1).log(),
                      'p_aug1': torch.clamp(p_aug1, 1e-7, 1).log(),
                      'p_aug2': torch.clamp(p_aug2, 1e-7, 1).log(),
                      'p_mixture': p_mixture}

    return loss, p_distribution


def jsdv1_3_1(pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None,
            **kwargs):
    """Calculate the jsdv1.1 loss.

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

    avg_factor = None
    temper = kwargs['temper']
    add_act = kwargs['add_act']

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)

    if pred_orig.shape[-1] == 1:  # if rpn
        # p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig), \
        #                           torch.sigmoid(pred_aug1),\
        #                           torch.sigmoid(pred_aug2)
        p_clean, p_aug1, p_aug2 = torch.cat((torch.sigmoid(pred_orig), 1 - torch.sigmoid(pred_orig)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug1), 1 - torch.sigmoid(pred_aug1)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug2), 1 - torch.sigmoid(pred_aug2)), dim=1),

    else:  # else roi
        p_clean, p_aug1, p_aug2 = F.softmax(pred_orig, dim=1), \
                                  F.softmax(pred_aug1, dim=1), \
                                  F.softmax(pred_aug2, dim=1)

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
        loss, weight=weight, reduction='sum', avg_factor=avg_factor)

    p_distribution = {'p_clean': torch.clamp(p_clean, 1e-7, 1).log(),
                      'p_aug1': torch.clamp(p_aug1, 1e-7, 1).log(),
                      'p_aug2': torch.clamp(p_aug2, 1e-7, 1).log(),
                      'p_mixture': p_mixture}

    return loss, p_distribution


def jsdv1_3_2(pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None,
            **kwargs):
    """Calculate the jsdv1.1 loss.

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

    avg_factor = None
    temper = kwargs['temper']
    add_act = kwargs['add_act']

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)

    if pred_orig.shape[-1] == 1:  # if rpn
        # p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig), \
        #                           torch.sigmoid(pred_aug1),\
        #                           torch.sigmoid(pred_aug2)
        p_clean, p_aug1, p_aug2 = torch.cat((torch.sigmoid(pred_orig), 1 - torch.sigmoid(pred_orig)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug1), 1 - torch.sigmoid(pred_aug1)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug2), 1 - torch.sigmoid(pred_aug2)), dim=1),

    else:  # else roi
        p_clean, p_aug1, p_aug2 = F.softmax(pred_orig, dim=1), \
                                  F.softmax(pred_aug1, dim=1), \
                                  F.softmax(pred_aug2, dim=1)

    p_clean, p_aug1, p_aug2 = p_clean.reshape((1,) + p_clean.shape).contiguous(), \
                              p_aug1.reshape((1,) + p_aug1.shape).contiguous(), \
                              p_aug2.reshape((1,) + p_aug2.shape).contiguous()

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    weight_orig = weight
    # apply weights and do the reduction
    if weight is not None:
        weight, _, _ = torch.chunk(weight, 3)
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    p_distribution = {'p_clean': torch.clamp(p_clean, 1e-7, 1).log(),
                      'p_aug1': torch.clamp(p_aug1, 1e-7, 1).log(),
                      'p_aug2': torch.clamp(p_aug2, 1e-7, 1).log(),
                      'p_mixture': p_mixture}

    """
    
    """
    ignore_index = kwargs['ignore_index']
    class_weight = kwargs['class_weight']
    lambda_weight = kwargs['lambda_weight']

    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight_orig, pred.size(-1),
                                              ignore_index)

    # weighted element-wise losses
    weight, _, _ = torch.chunk(weight, 3)
    if weight is not None:
        weight = weight.float()

    label, _, _ = torch.chunk(label, 3)

    loss_aug1 = F.binary_cross_entropy_with_logits(
        pred_aug1, label.float(), pos_weight=class_weight, reduction='none')
    loss_aug2 = F.binary_cross_entropy_with_logits(
        pred_aug2, label.float(), pos_weight=class_weight, reduction='none')

    # do the reduction for the weighted loss
    loss_aug1 = weight_reduce_loss(
        loss_aug1, weight, reduction=reduction, avg_factor=avg_factor)
    loss_aug2 = weight_reduce_loss(
        loss_aug2, weight, reduction=reduction, avg_factor=avg_factor)

    loss_orig = loss_aug1 + loss_aug2

    return loss + (loss_orig / lambda_weight), p_distribution

def jsdv1_3_3(pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None,
            **kwargs):
    """Calculate the jsdv1.1 loss.

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

    avg_factor = None
    temper = kwargs['temper']
    add_act = kwargs['add_act']

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)

    if pred_orig.shape[-1] == 1:  # if rpn
        # p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig), \
        #                           torch.sigmoid(pred_aug1),\
        #                           torch.sigmoid(pred_aug2)
        p_clean, p_aug1, p_aug2 = torch.cat((torch.sigmoid(pred_orig), 1 - torch.sigmoid(pred_orig)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug1), 1 - torch.sigmoid(pred_aug1)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug2), 1 - torch.sigmoid(pred_aug2)), dim=1),

    else:  # else roi
        p_clean, p_aug1, p_aug2 = F.softmax(pred_orig, dim=1), \
                                  F.softmax(pred_aug1, dim=1), \
                                  F.softmax(pred_aug2, dim=1)

    p_clean, p_aug1, p_aug2 = p_clean.reshape((1,) + p_clean.shape).contiguous(), \
                              p_aug1.reshape((1,) + p_aug1.shape).contiguous(), \
                              p_aug2.reshape((1,) + p_aug2.shape).contiguous()

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    p_clean = torch.clamp(p_clean, 1e-7, 1)
    p_aug1 = torch.clamp(p_aug1, 1e-7, 1)
    p_aug2 = torch.clamp(p_aug2, 1e-7, 1)

    loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    # apply weights and do the reduction
    if weight is not None:
        weight, _, _ = torch.chunk(weight, 3)
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    p_distribution = {'p_clean': torch.clamp(p_clean, 1e-7, 1).log(),
                      'p_aug1': torch.clamp(p_aug1, 1e-7, 1).log(),
                      'p_aug2': torch.clamp(p_aug2, 1e-7, 1).log(),
                      'p_mixture': p_mixture}

    loss = assert_positive_loss(loss, type='jsdv1.3.3',
                                pred_orig=pred_orig, pred_aug1=pred_aug1, pred_aug2=pred_aug2,
                                weight=weight)

    return loss, p_distribution


def jsdv1_3_4(pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None,
            **kwargs):
    """Calculate the jsdv1.1 loss.

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

    avg_factor = None
    temper = kwargs['temper']
    add_act = kwargs['add_act']

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)

    if pred_orig.shape[-1] == 1:  # if rpn
        # p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig), \
        #                           torch.sigmoid(pred_aug1),\
        #                           torch.sigmoid(pred_aug2)
        p_clean, p_aug1, p_aug2 = torch.cat((torch.sigmoid(pred_orig), 1 - torch.sigmoid(pred_orig)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug1), 1 - torch.sigmoid(pred_aug1)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug2), 1 - torch.sigmoid(pred_aug2)), dim=1),

    else:  # else roi
        p_clean, p_aug1, p_aug2 = F.softmax(pred_orig, dim=1), \
                                  F.softmax(pred_aug1, dim=1), \
                                  F.softmax(pred_aug2, dim=1)

    p_clean, p_aug1, p_aug2 = p_clean.reshape((1,) + p_clean.shape).contiguous(), \
                              p_aug1.reshape((1,) + p_aug1.shape).contiguous(), \
                              p_aug2.reshape((1,) + p_aug2.shape).contiguous()

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction='mean') +
            F.kl_div(p_mixture, p_aug1, reduction='mean') +
            F.kl_div(p_mixture, p_aug2, reduction='mean')) / 3.

    # apply weights and do the reduction
    if weight is not None:
        weight, _, _ = torch.chunk(weight, 3)
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    p_distribution = {'p_clean': torch.clamp(p_clean, 1e-7, 1).log(),
                      'p_aug1': torch.clamp(p_aug1, 1e-7, 1).log(),
                      'p_aug2': torch.clamp(p_aug2, 1e-7, 1).log(),
                      'p_mixture': p_mixture}

    return loss, p_distribution


def jsdv1_3_5(pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None,
            **kwargs):
    """Calculate the jsdv1.3 loss.
    jsd loss (sigmoid, 1-sigmoid) for rpn head, softmax for roi head
    divided by batchmean, divided by 768 (256*3) for rpn, 1056 (352*3) for roi
    reduction parameter does not affect the loss
    jsdv_inversion version

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

    # avg_factor = None
    temper = kwargs['temper']
    add_act = kwargs['add_act']

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)

    if pred_orig.shape[-1] == 1:  # if rpn
        # p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig), \
        #                           torch.sigmoid(pred_aug1),\
        #                           torch.sigmoid(pred_aug2)
        p_clean, p_aug1, p_aug2 = torch.cat((torch.sigmoid(pred_orig), 1 - torch.sigmoid(pred_orig)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug1), 1 - torch.sigmoid(pred_aug1)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug2), 1 - torch.sigmoid(pred_aug2)), dim=1),

    else:  # else roi
        p_clean, p_aug1, p_aug2 = F.softmax(pred_orig, dim=1), \
                                  F.softmax(pred_aug1, dim=1), \
                                  F.softmax(pred_aug2, dim=1)

    p_clean, p_aug1, p_aug2 = p_clean.reshape((1,) + p_clean.shape).contiguous(), \
                              p_aug1.reshape((1,) + p_aug1.shape).contiguous(), \
                              p_aug2.reshape((1,) + p_aug2.shape).contiguous()

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = (p_clean + p_aug1 + p_aug2) / 3
    p_clean = torch.clamp(p_clean, 1e-7, 1).log()
    p_aug1 = torch.clamp(p_aug1, 1e-7, 1).log()
    p_aug2 = torch.clamp(p_aug2, 1e-7, 1).log()

    loss = (F.kl_div(p_clean, p_mixture, reduction='batchmean') +
            F.kl_div(p_aug1, p_mixture, reduction='batchmean') +
            F.kl_div(p_aug2, p_mixture, reduction='batchmean')) / 3.

    # apply weights and do the reduction
    if weight is not None:
        weight, _, _ = torch.chunk(weight, 3)
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    p_distribution = {'p_clean': p_clean,
                      'p_aug1': p_aug1,
                      'p_aug2':p_aug2,
                      'p_mixture': torch.clamp(p_mixture, 1e-7, 1).log()}

    return loss, p_distribution


def jsdv1_4(pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None,
            **kwargs):
    """Calculate the jsdv1.1 loss.

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

    avg_factor = None
    kldiv_reduction = 'batchmean'

    temper = kwargs['temper']

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)

    if pred_orig.shape[-1] == 1:  # if rpn
        # p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig), \
        #                           torch.sigmoid(pred_aug1),\
        #                           torch.sigmoid(pred_aug2)
        p_clean, p_aug1, p_aug2 = torch.cat((torch.sigmoid(pred_orig / temper), 1 - torch.sigmoid(pred_orig / temper)),
                                            dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug1 / temper), 1 - torch.sigmoid(pred_aug1 / temper)),
                                            dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug2 / temper), 1 - torch.sigmoid(pred_aug2 / temper)),
                                            dim=1),

    else:  # else roi
        p_clean, p_aug1, p_aug2 = F.softmax(pred_orig / temper, dim=1), \
                                  F.softmax(pred_aug1 / temper, dim=1), \
                                  F.softmax(pred_aug2 / temper, dim=1)

    p_clean, p_aug1, p_aug2 = p_clean.reshape((1,) + p_clean.shape).contiguous(), \
                              p_aug1.reshape((1,) + p_aug1.shape).contiguous(), \
                              p_aug2.reshape((1,) + p_aug2.shape).contiguous()

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction=kldiv_reduction) +
            F.kl_div(p_mixture, p_aug1, reduction=kldiv_reduction) +
            F.kl_div(p_mixture, p_aug2, reduction=kldiv_reduction)) / 3.

    # apply weights and do the reduction
    if weight is not None:
        weight, _, _ = torch.chunk(weight, 3)
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    p_distribution = {'p_clean': torch.clamp(p_clean, 1e-7, 1).log(),
                      'p_aug1': torch.clamp(p_aug1, 1e-7, 1).log(),
                      'p_aug2': torch.clamp(p_aug2, 1e-7, 1).log(),
                      'p_mixture': p_mixture}

    return loss, p_distribution


def jsdy(pred,
         label,
         weight=None,
         reduction='mean',
         avg_factor=None,
         **kwargs):
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
        if pred_orig.shape[-1] == 1:  # if rpn
            label = label.reshape(label.shape + (1,)).contiguous()
        else:  # else roi
            label = F.one_hot(label, num_classes=pred_orig.shape[-1])  # TO-DO: need to check

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
                      'p_mixture': p_mixture,
                      'label': label}

    return loss, p_distribution


def jsdv2(pred,
          label,
          weight=None,
          reduction='mean',
          avg_factor=None,
          ignore_index=-100,
          **kwargs):
    """Calculate the jsdy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        temper (int, optional): temperature scaling for softmax function.

    Returns:
        torch.Tensor: The calculated loss
    """
    temper = kwargs['temper']
    add_act = kwargs['add_act']

    # chunk the data to get p_orig, label_orig, and weight_orig
    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    label, _, _ = torch.chunk(label, 3)
    if weight is not None:
        weight, _, _ = torch.chunk(weight, 3)
        weight = weight.float()

    # match the shape: label and weight with pred
    if pred_orig.shape != label.shape:
        if pred_orig.shape[-1] == 1:  # if rpn
            ignore_index = -100 if ignore_index is None else ignore_index
            label, weight = _expand_onehot_labels(label, weight, pred_orig.size(-1), ignore_index)  # label conversion
        else:  # else roi
            # label_ = F.one_hot(label, num_classes=pred_orig.shape[-1])  # deprecated
            label, weight = _expand_onehot_labels(label, weight, pred_orig.size(-1), ignore_index)  # same as F.one_hot

    if add_act == None:
        # sigmoid and softmax function for rpn_cls and roi_cls
        if pred_orig.shape[-1] == 1:  # if rpn
            p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig / temper), \
                                      torch.sigmoid(pred_aug1 / temper), \
                                      torch.sigmoid(pred_aug2 / temper)
        else:  # else roi
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

    label = label.float()

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction='none') +
            F.kl_div(p_mixture, p_aug1, reduction='none') +
            F.kl_div(p_mixture, p_aug2, reduction='none')) / 3.

    # apply weights and do the reduction
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=None)  # avg_factor=avg_factor is deprecated

    if weight is not None:
        assert p_clean.size() == label.size() == weight.size(), \
            "The size of tensors does not match"
        # get valid predictions for wandb log
        p_clean, p_aug1, p_aug2, p_mixture, label = torch.clamp(p_clean[weight != 0], 1e-7, 1).log(), \
                                                    torch.clamp(p_aug1[weight != 0], 1e-7, 1).log(), \
                                                    torch.clamp(p_aug2[weight != 0], 1e-7, 1).log(), \
                                                    p_mixture[weight != 0], \
                                                    torch.clamp(label[weight != 0], 1e-7, 1).log()

    # logging
    p_distribution = {'p_clean': p_clean,
                      'p_aug1': p_aug1,
                      'p_aug2': p_aug2,
                      'p_mixture': p_mixture,
                      'label': label, }

    return loss, p_distribution


def jsdv3(pred,
          label,
          weight=None,
          reduction='mean',
          avg_factor=None,
          ignore_index=-100,
          **kwargs):
    """Calculate the jsdy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        temper (int, optional): temperature scaling for softmax function.

    Returns:
        torch.Tensor: The calculated loss
    """
    temper = kwargs['temper']
    add_act = kwargs['add_act']

    # chunk the data to get p_orig, label_orig, and weight_orig
    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    label, _, _ = torch.chunk(label, 3)
    if weight is not None:
        weight, _, _ = torch.chunk(weight, 3)
        weight = weight.float()

    # match the shape: label and weight with pred
    if pred_orig.shape != label.shape:
        if pred_orig.shape[-1] == 1:  # if rpn
            ignore_index = -100 if ignore_index is None else ignore_index
            label, weight = _expand_onehot_labels(label, weight, pred_orig.size(-1), ignore_index)  # label conversion
        else:  # else roi
            # label_ = F.one_hot(label, num_classes=pred_orig.shape[-1])  # deprecated
            label, weight = _expand_onehot_labels(label, weight, pred_orig.size(-1), ignore_index)  # same as F.one_hot

    if add_act == None:
        # sigmoid and softmax function for rpn_cls and roi_cls
        if pred_orig.shape[-1] == 1:  # if rpn
            p_clean, p_aug1, p_aug2 = torch.sigmoid(pred_orig / temper), \
                                      torch.sigmoid(pred_aug1 / temper), \
                                      torch.sigmoid(pred_aug2 / temper)
        else:  # else roi
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

    label = label.float()

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    # apply weights and do the reduction
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=None)  # avg_factor=avg_factor is deprecated

    if weight is not None:
        assert p_clean.size() == label.size() == weight.size(), \
            "The size of tensors does not match"
        # get valid predictions for wandb log
        p_clean, p_aug1, p_aug2, p_mixture, label = torch.clamp(p_clean[weight != 0], 1e-7, 1).log(), \
                                                    torch.clamp(p_aug1[weight != 0], 1e-7, 1).log(), \
                                                    torch.clamp(p_aug2[weight != 0], 1e-7, 1).log(), \
                                                    p_mixture[weight != 0], \
                                                    torch.clamp(label[weight != 0], 1e-7, 1).log()

    # logging
    p_distribution = {'p_clean': p_clean,
                      'p_aug1': p_aug1,
                      'p_aug2': p_aug2,
                      'p_mixture': p_mixture,
                      'label': label, }

    return loss, p_distribution


def assert_positive_loss(loss, **kwargs):
    def print_kwargs(keyword, save=False):
        if keyword in kwargs:
            obj = kwargs[keyword]
            if save:
                path = f'{filename}_{keyword}.pt'
                torch.save(obj, path)
                if isinstance(obj, dict):
                    f.write(f"{keyword} {obj.shape}: {path}\n")
                elif isinstance(obj, list):
                    f.write(f"{keyword} {len(obj)}: {path}\n")
                elif torch.is_tensor(obj):
                    f.write(f"{keyword} : {path}\n")
            else:
                f.write(f"{keyword} : {obj}\n")
    try:
        if loss <= 0:
            raise ValueError
        else:
            return loss
    except ValueError:
        import time
        if loss == 0:
            filename = f"/ws/data/dshong/invalid_jsd_value/{time.strftime('%Y%m%d_%H%M%S')}"
            f = open(f"{filename}.txt", 'w')
            f.write(f"[1] Information \n")
            print_kwargs('type')
            print_kwargs('name')
            f.write("\n\n")

            f.write("[2] Value\n")
            print_kwargs('pred_orig', True)
            print_kwargs('label', True)
            print_kwargs('weight', True)
            print_kwargs('pred_aug1', True)
            print_kwargs('pred_aug2', True)
            print_kwargs('class_weight', True)

            f.close()
        else:
            filename = f"/ws/data/dshong/invalid_jsd_value/negative_loss"
            f = open(f"{filename}.txt", 'a')
            f.write(f"{time.strftime('%Y%m%d_%H%M%S')}")
            f.close()
        return 0


@LOSSES.register_module()
class CrossEntropyLossPlus(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 additional_loss='jsd',
                 additional_loss_weight_reduce=False,
                 lambda_weight=0.0001,
                 temper=1,
                 add_act=None,
                 wandb_name=None,
                 **kwargs):
        """CrossEntropyLossPlus.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            temper (int, optional): temperature scaling for softmax function.
        """
        super(CrossEntropyLossPlus, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.additional_loss = additional_loss
        self.additional_loss_weight_reduce = additional_loss_weight_reduce
        self.lambda_weight = lambda_weight
        self.temper = temper
        self.add_act = add_act
        self.wandb_name = wandb_name

        self.kwargs = kwargs

        self.wandb_features = dict()
        self.wandb_features[f'additional_loss({self.wandb_name})'] = []
        self.wandb_features[f'ce_loss({self.wandb_name})'] = []

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        if self.additional_loss == 'jsd':
            self.cls_additional = jsd
        elif self.additional_loss == 'jsdv1_1':
            self.cls_additional = jsdv1_1
        elif self.additional_loss == 'jsdv1_2':
            self.cls_additional = jsdv1_2
        elif self.additional_loss == 'jsdv1_3':
            self.cls_additional = jsdv1_3
        elif self.additional_loss == 'jsdv1_3_1':
            self.cls_additional = jsdv1_3_1
        elif self.additional_loss == 'jsdv1_3_2':
            self.cls_additional = jsdv1_3_2
        elif self.additional_loss == 'jsdv1_3_3':
            self.cls_additional = jsdv1_3_3
        elif self.additional_loss == 'jsdv1_3_4':
            self.cls_additional = jsdv1_3_4
        elif self.additional_loss == 'jsdv1_3_5':
            self.cls_additional = jsdv1_3_5
        elif self.additional_loss == 'jsdv1_4':
            self.cls_additional = jsdv1_4
        elif self.additional_loss == 'jsdv2':
            self.cls_additional = jsdv2
        elif self.additional_loss == 'jsdv3':
            self.cls_additional = jsdv3
        elif self.additional_loss == 'jsdy':
            self.cls_additional = jsdy
        else:
            self.cls_additional = None

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=kwargs['original_avg_factor'] if 'original_avg_factor' in kwargs else avg_factor,
            ignore_index=ignore_index)

        loss_additional = 0
        if self.cls_additional is not None:
            if self.additional_loss_weight_reduce == False:
                weight = None
            loss_additional, p_distribution = self.cls_additional(
                cls_score,
                label,
                weight,
                reduction=reduction,
                avg_factor=avg_factor,
                temper=self.temper,
                add_act=self.add_act,
                ignore_index=ignore_index,
                class_weight=class_weight,
                lambda_weight=self.lambda_weight
                )

            # wandb for rpn
            if self.use_sigmoid:
                if len(self.wandb_features[f'ce_loss({self.wandb_name})']) == 5:
                    self.wandb_features[f'ce_loss({self.wandb_name})'].clear()
                    self.wandb_features[f'additional_loss({self.wandb_name})'].clear()
                self.wandb_features[f'ce_loss({self.wandb_name})'].append(loss_cls)
                self.wandb_features[f'additional_loss({self.wandb_name})'].append(
                    self.lambda_weight * loss_additional)
            else:
                self.wandb_features[f'ce_loss({self.wandb_name})'] = loss_cls
                self.wandb_features[f'additional_loss({self.wandb_name})'] = self.lambda_weight * loss_additional

            for key, value in p_distribution.items():
                self.wandb_features[f'{key}({self.wandb_name})'] = value

        loss = loss_cls + self.lambda_weight * loss_additional
        # self.wandb_features[f'loss({self.wandb_name})'] = loss
        # self.wandb_features[f'additional_loss({self.wandb_name})'] = loss_additional
        return loss

