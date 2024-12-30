from ast import main
from codecs import ignore_errors

import torch
import torch.nn as nn

import builder
import dataset
import model
import utils
from engine.finetune_engine import FinetuneEngine
from engine.linear_eval_engine import LinearEvalEngine
from engine.pretraining_engine import PretrainingEngine
from engine.train_val_test import train
from engine.training_engine import SupervisedTrainingEngine

torch.backends.cudnn.enable = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


with utils.log.TimerBlock("start") as block:
    """
    Here is the script to train the model.
    """
    utils.init_seeds()

    args = utils.parser_args.parser_args(block)

    # 设置模型 数据集 优化器 损失函数

    ## MMFi的数据处理太过麻烦没有直接用注册器生成实例
    if args.data_param["type"] == "MMFi":
        import dataset.mmfi as mmfi

        config = args.data_param
        dataset_root = config["dataset_root"]
        train_dataset, val_dataset = mmfi.make_dataset(dataset_root, config)
        rng_generator = torch.manual_seed(config["init_rand_seed"])
        train_loader = mmfi.make_dataloader(
            train_dataset, is_training=True, generator=rng_generator, **config["train_loader"]
        )
        val_loader = mmfi.make_dataloader(
            val_dataset, is_training=False, generator=rng_generator, **config["validation_loader"]
        )
    else:
        dataloaders = builder.build_dataset(cfg=args.data_param, block=block, registry=builder.DatasetRegistry)
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]

    if args.run_mode == "train_val":
        model = builder.build_model(cfg=args.model_param, registry=builder.ModelRegistry)

        # load checkpoint might not work, which has not been tested.
        utils.load_checkpoint.load_from_pretrained_model(model, block, args)

        head = builder.build_head(cfg=args.head_param, registry=builder.HeadRegistry)

        optimizer = builder.build_optimizer(model, head, block=block, **args.optimizer_param)

        loss_function = builder.build_loss(args.loss)

        model.cuda()
        head.cuda()
        model = nn.DataParallel(model, device_ids=args.device_id)
        head = nn.DataParallel(head, device_ids=args.device_id)
        block.log("copy model to gpu")

        engine = SupervisedTrainingEngine(
            block=block,
            model=model,
            head=head,
            optimizer=optimizer,
            loss_function=loss_function,
            train_loader=train_loader,
            val_loader=val_loader,
            model_save_path=args.output_path,
            **args.training_param,
        )

    elif args.run_mode == "pretrain":
        from model.backbone.resnet import resnet_backbone
        from pretrain.method.simclr import SimCLR

        backbone = resnet_backbone(model_name="resnet18", weights=None)
        method = SimCLR(backbone=backbone, mode="train", **args.augmentation)
        engine = PretrainingEngine(
            block=block,
            method=method,
            train_loader=train_loader,
            val_loader=val_loader,
            model_save_path=args.output_path,
            pretrained_model=args.pretrained_model,
            ignore_weights=args.ignore_weights,
            **args.training_param,
        )

    elif args.run_mode == "linear_eval":
        from model.backbone.resnet import resnet_backbone
        from pretrain.method.simclr import SimCLR

        loss_function = builder.build_loss(args.loss)
        backbone = resnet_backbone(model_name="resnet18", weights=None)
        method = SimCLR(backbone=backbone, mode="linear_eval", criterion=loss_function, **args.augmentation)
        engine = LinearEvalEngine(
            block=block,
            method=method,
            train_loader=train_loader,
            val_loader=val_loader,
            model_save_path=args.output_path,
            pretrained_model=args.pretrained_model,
            ignore_weights=args.ignore_weights,
            **args.training_param,
        )

    elif args.run_mode == "finetune":
        from model.backbone.resnet import resnet_backbone
        from pretrain.method.simclr import SimCLR

        loss_function = builder.build_loss(args.loss)
        backbone = resnet_backbone(model_name="resnet18", weights=None)
        method = SimCLR(backbone=backbone, mode="finetune", criterion=loss_function, **args.augmentation)
        engine = FinetuneEngine(
            block=block,
            method=method,
            train_loader=train_loader,
            val_loader=val_loader,
            model_save_path=args.output_path,
            pretrained_model=args.pretrained_model,
            ignore_weights=args.ignore_weights,
            **args.training_param,
        )

    assert args.run_mode in engine.run_modes
    getattr(engine, args.run_mode, None)()

if __name__ == "__main__":
    pass
