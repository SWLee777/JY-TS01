#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/9/28 下午12:13
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : init_func.py.py
import torch
import torch.nn as nn

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    """Create exhaustive AdamW groups, including parameters owned directly.

    The previous implementation only recognized weights inside Linear/Conv/Norm
    modules.  VMamba keeps several trainable tensors directly on modules and the
    residual decoder keeps its zero gates in a ParameterDict, so those tensors
    were silently absent from the optimizer.
    """
    group_decay = []
    group_no_decay = []
    seen = set()
    norm_types = (
        norm_layer,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.GroupNorm,
        nn.LayerNorm,
    )

    for _, owner in module.named_modules():
        owner_is_norm = isinstance(owner, norm_types)
        for parameter_name, parameter in owner.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            no_decay = (
                owner_is_norm
                or parameter_name.endswith("bias")
                or parameter.ndim <= 1
                or getattr(parameter, "_no_weight_decay", False)
            )
            (group_no_decay if no_decay else group_decay).append(parameter)

    expected = {
        id(parameter)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    if seen != expected:
        missing = len(expected - seen)
        duplicated_or_extra = len(seen - expected)
        raise RuntimeError(
            "Optimizer grouping is incomplete: "
            f"missing={missing}, extra={duplicated_or_extra}."
        )

    if group_decay:
        weight_group.append(dict(params=group_decay, lr=lr))
    if group_no_decay:
        weight_group.append(
            dict(params=group_no_decay, weight_decay=0.0, lr=lr)
        )
    return weight_group
