import torch.nn as nn
import torch
from builder.builder import ModelRegistry, DatasetRegistry, HeadRegistry
from builder.builder import build_model, build_dataset, build_loss, build_optimizer, build_head
from engine.training_engine import TrainingEngine
from utils import parser_args
from utils.base_util import init_seeds
from utils.load_checkpoint import load_from_pretrained_model
from utils.log import TimerBlock
import dataset.mmfi as mmfi

torch.backends.cudnn.enable = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
"""
Importing these module is important because of the registry, although they seem like useless.
You need to import the module which include the model network that you wanna use.   
"""
from model.sensefi import UT_HAR_model
from model.skl import stgcn, stgat
from model.that import that
from model.KAN import kan
from dataset import ntu_fi
from dataset import ut_har_dataset
from model.heads import simple_head

with TimerBlock("start") as block:
    """
    Here is the script to train the model.
    """
    init_seeds()

    args = parser_args.parser_args(block)

    # 设置模型 数据集 优化器 损失函数
    model = build_model(cfg=args.model_param, registry=ModelRegistry)

    # load checkpoint might not work, which has not been tested.
    load_from_pretrained_model(model, block, args)

    ## MMFi的数据处理太过麻烦没有直接用注册器生成实例
    if args.data_param["type"] == "MMFi":
        config = args.data_param
        dataset_root = config["dataset_root"]
        train_dataset, val_dataset = mmfi.make_dataset(dataset_root, config)
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = mmfi.make_dataloader(train_dataset, is_training=True, generator=rng_generator,
                                            **config['train_loader'])
        val_loader = mmfi.make_dataloader(val_dataset, is_training=False, generator=rng_generator,
                                          **config['validation_loader'])
    else:
        dataloaders = build_dataset(cfg=args.data_param, block=block, registry=DatasetRegistry)
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']

    # signal_pipeline = build_signal_pipeline()

    head = build_head(cfg=args.head_param, registry=HeadRegistry)

    optimizer = build_optimizer(model, head, block=block, **args.optimizer_param)

    loss_function = build_loss(args.loss)

    model.cuda()
    head.cuda()
    model = nn.DataParallel(model, device_ids=args.device_id)
    head = nn.DataParallel(head, device_ids=args.device_id)
    block.log('copy model to gpu')

    engine = TrainingEngine(block=block, model=model, head=head, optimizer=optimizer, loss_function=loss_function,
                            train_loader=train_loader, val_loader=val_loader, model_save_path=args.output_path,
                            **args.training_param)

    assert args.run_mode in engine.run_modes
    getattr(engine, args.run_mode, None)()
