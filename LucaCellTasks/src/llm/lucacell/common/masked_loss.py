#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/6/28 10:25
@project: LucaOne
@file: masked_loss
@desc: masked loss for LucaOne
'''
import torch
import torch.nn as nn


class _MaskedLoss(nn.Module):
    """Base class for masked losses"""

    def __init__(self, reduction='mean', ignore_nans=True, ignore_value=-100.0):
        super().__init__()
        self.reduction = reduction
        self.ignore_nans = ignore_nans
        self.ignore_value = ignore_value

    def forward(self, pred, target, mask=None):
        """Compute a loss between pred and target for given mask.
        Note that this implementation is faster than loss(pred[mask], target[mask])
        for a given loss, and is nan-proof."""
        '''
        if not (target.size() == pred.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the pred size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    target.size(), pred.size()),
                stacklevel=2,
            )
        '''
        if mask is None and self.ignore_value is not None:
            mask = target != self.ignore_value
        elif mask is None:
            mask = torch.ones_like(target, dtype=bool)
        target_proxy = target
        if self.ignore_nans:
            target_proxy = target.clone()
            nans = torch.isnan(target)
            if nans.any():
                with torch.no_grad():
                    mask = mask & ~nans
                    target_proxy[nans] = 0
        # full_loss = self.criterion(pred, target_proxy)
        # print("mask shape")
        # print(mask.shape)
        if self.reduction == 'meanmean' and pred.ndim == 3 and pred.shape[-1] == 1:
            # gene_level binary classification
            # pred: n , seq_len, 1 -> n * seq_len
            # target: n, seq_len -> n * seq_len
            full_loss = self.criterion(pred.view(-1), target_proxy.view(-1))
            full_loss = torch.reshape(full_loss, (-1, pred.shape[1]))
        elif self.reduction == 'meanmean' and pred.ndim == 3:
            if target.ndim == 3:
                # gene_level regression
                # pred: n , seq_len, label_size -> n * seq_len * label_size
                # target: n, seq_len, label_size -> n * seq_len * label_size
                full_loss = self.criterion(pred.view(-1), target_proxy.view(-1))
                full_loss = torch.reshape(full_loss, (-1, pred.shape[1], pred.shape[-1]))
            else:
                # gene_level multi classification
                # pred: n , seq_len, label_size -> n * seq_len, label_size
                # target: n, seq_len -> n * seq_len
                full_loss = self.criterion(pred.view(-1, pred.shape[-1]), target_proxy.view(-1))
                full_loss = torch.reshape(full_loss, (-1, pred.shape[1]))
        elif self.reduction == 'meanmean' and pred.ndim == 2 and target.ndim == 2:
            # cell-level multi label
            # pred: n , label_size -> n * label_size
            # target: n, label_size -> n * label_size
            full_loss = self.criterion(pred.view(-1), target_proxy.view(-1))
            full_loss = torch.reshape(full_loss, (-1, pred.shape[1]))
        elif self.reduction == 'meanmean':
            self.reduction = "mean"
            full_loss = self.criterion(pred, target_proxy)
        else:
            full_loss = self.criterion(pred, target_proxy)

        full_loss[~mask] = 0
        if self.reduction == 'none':
            return full_loss
        if self.reduction == 'sum':
            return full_loss.sum()
        if self.reduction == 'mean':
            return full_loss.sum() / (mask.to(full_loss.dtype).sum() + 1e-12)
        if self.reduction == 'meanmean':
            if mask.ndim == 3:
                mask_sum = mask.to(full_loss.dtype).sum(dim=-1)
                full_loss = full_loss.sum(dim=-1) / (mask_sum + 1e-12)
                mask_sum = mask_sum.to(torch.bool).sum(dim=-1)
                # print(mask_sum)
                full_loss = full_loss.sum(dim=-1) / (mask_sum + 1e-12)
                mask_sum = mask_sum.to(torch.bool).sum()
                # print(mask_sum)
                loss = full_loss.sum() / (mask_sum + 1e-12)
            else:
                mask_sum = mask.to(full_loss.dtype).sum(dim=-1)
                loss = torch.sum(full_loss.sum(dim=-1) / (mask_sum + 1e-12)) / (mask_sum.to(torch.bool).sum() + 1e-12)
            # print(full_loss.sum() / (mask.to(full_loss.dtype).sum() + 1e-12), loss)
            return loss
        if self.reduction in ["summean", "meansum"]:
            if mask.ndim == 3:
                mask_sum = mask.to(full_loss.dtype).sum(dim=-1)
                full_loss = full_loss.sum(dim=-1)
                mask_sum = mask_sum.to(torch.bool).sum(dim=-1)
                # print(mask_sum)
                full_loss = full_loss.sum(dim=-1) / (mask_sum + 1e-12)
                mask_sum = mask_sum.to(torch.bool).sum()
                # print(mask_sum)
                loss = full_loss.sum() / (mask_sum + 1e-12)
            else:
                mask_sum = mask.to(full_loss.dtype).sum(dim=-1)
                loss = full_loss.sum() / (mask_sum.to(torch.bool).sum() + 1e-12)
            return loss
        return full_loss


