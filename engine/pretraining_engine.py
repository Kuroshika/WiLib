import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model
from utils.log import IteratorTimer

from .training_engine import TrainingEngine


class PretrainingEngine(TrainingEngine):
    def __init__(
        self,
        method,
        train_loader,
        max_epoch,
        dataloader_datatype,
        model_datatype,
        block=None,
        val_loader=None,
        test_loader=None,
        model_save_path=None,
        save_frequency: int = 1,
        val_frequency: int = 1,
    ):
        super().__init__()
        self.method = method

        self.save_frequency = save_frequency
        self.dataloader_datatype = dataloader_datatype
        self.model_datatype = model_datatype

        self.block = block
        self.val_frequency = val_frequency

        self.memory = {}
        assert model_save_path is not None
        self.model_save_path = model_save_path

        self.max_epoch = max_epoch
        self.epoch_now = 0

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.summary_writer = SummaryWriter(self.model_save_path)

        self.run_modes = ("pretrain",)

    def pretrain(self, val: bool = False):
        for epoch in range(self.max_epoch):
            self.epoch_now = epoch
            total_loss = 0
            self.log(f"Begin to train for epoch[{epoch}]")
            self.pretrain_epoch(epoch)
            # if (epoch + 1) % self.val_frequency == 0:
            #     self.log(f"Begin to val for the pretrain epoch[{epoch}]:")
            #     log_info = self.val_epoch(epoch)
            #     self.log(log_info)
            #     if log_info["is_best"]:
            #         self.method.save_model(is_best=True, epoch=epoch, model_save_path=self.model_save_path)
            self.log(f"Finish training for epoch[{epoch}]")

        if (epoch + 1) % self.save_frequency == 0:
            self.method.save_model(is_best=False, epoch=epoch, model_save_path=self.model_save_path)

    def pretrain_epoch(self, epoch):
        loss_total = 0
        step = 0
        process = tqdm(IteratorTimer(self.train_loader), desc="Train: ")
        for index, (inputs, labels) in enumerate(process):
            inputs = self.convert_to_tensor(inputs)
            inputs = inputs.to(self.device)
            inputs = self.align_data_with_model(inputs)
            # 修改该部分Method的传播逻辑
            ls = self.method.train_step(inputs)
            loss_total = loss_total + ls
            step += 1
            process.set_description(
                f"Train: epoch:{epoch} loss: {ls:4f}, batch time: {process.iterable.last_duration:4f}"
            )
        process.close()
        loss = loss_total / step
        self.summary_writer.add_scalar("train_loss", loss, epoch)
