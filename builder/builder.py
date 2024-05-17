import inspect
import random

import numpy as np
import torch
import shutil
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD
from builder.registry import Registry

import argparse

from model import *

ModelRegistry = Registry('model')
OptimRegistry = Registry('optim')
DatasetRegistry = Registry('dataset')
HeadRegistry = Registry('head')


def init_seed(x):
    # pass
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg, registry, default_args=None):
    args = cfg.copy()
    obj_type = args.pop('type')  # 从配置文件中索引出type字段对应的obj
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)  # 根据字段从Registry中索引出类
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')

    return obj_cls(**args)  # 完成类的初始化


def build_dataset(cfg, block, registry):
    args = cfg.copy()
    data_type = args.pop('type')

    if isinstance(data_type, str):
        data_cls = registry.get(data_type)  # 根据字段从Registry中索引出类
        if data_cls is None:
            raise KeyError(
                f'{data_type} is not in the {registry.name} registry')
    elif inspect.isclass(data_type):
        data_cls = data_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(data_type)}')

    batch_size = args.pop("batch_size")
    workers = args.pop("workers")
    pin_memory = args.pop("pin_memory")
    dataset_mode = args.pop("dataset_mode")
    drop_last = args.pop("drop_last")

    dataloader_dict = {}
    for mode in dataset_mode:
        assert mode in ['train', "val", "test"], "dataset mode should be in ('train','val','test')"
        dataset = data_cls(mode=mode, **args[f'{mode}_data_param'])
        is_shuffle = bool(mode == 'train')
        dataloader_dict[mode] = DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=is_shuffle,
                                           num_workers=workers,
                                           drop_last=drop_last,
                                           pin_memory=pin_memory,
                                           worker_init_fn=init_seed,
                                           )
    return dataloader_dict


def build_loss(loss_name='cross_entropy'):
    loss_function = None
    if loss_name == 'cross_entropy':
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        raise Exception(f"The loss function [{loss_name}] is not supported!")

    return loss_function


def build_optimizer(model, head, optim_type, block, lr, wd):
    params = []
    for key, value in model.named_parameters():
        if value.requires_grad:
            params += [{'params': [value], 'lr': lr, 'key': key, 'weight_decay': wd}]
    for key, value in head.named_parameters():
        if value.requires_grad:
            params += [{'params': [value], 'lr': lr, 'key': key, 'weight_decay': wd}]
    if optim_type == 'adam':
        optim_type = torch.optim.Adam(params, betas=(0.9, 0.999))
        block.log('Using Adam optimizer')
    elif optim_type == 'adamw':
        optim_type = torch.optim.AdamW(params)
        block.log('Using AdamW optimizer')
    elif optim_type == 'sgd':
        momentum = 0.9
        optim_type = SGD(params, momentum=momentum)
        block.log('Using SGD with momentum ' + str(momentum))
    elif optim_type == 'sgd_nev':
        momentum = 0.9
        optim_type = SGD(params, momentum=momentum, nesterov=True)
        block.log('Using SGD with momentum ' + str(momentum) + 'and nesterov')
    else:
        momentum = 0.9
        optim_type = SGD(params, momentum=momentum)
        block.log('Using SGD with momentum ' + str(momentum))

    # shutil.copy2(inspect.getfile(optimizer), args.model_saved_name)
    block.copy2out(__file__)
    return optim_type

def build_head(cfg, registry, default_args=None):
    args = cfg.copy()
    obj_type = args.pop('type')  # 从配置文件中索引出type字段对应的obj

    if obj_type == "no_head":
        return None

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)  # 根据字段从Registry中索引出类
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')

    return obj_cls(**args)  # 完成类的初始化