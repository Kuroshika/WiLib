# from typing import Tuple

# from ast import Tuple
import os
import time

import torch
from numpy import dtype
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model
from engine.train_classifier import val_classifier
from utils.load_checkpoint import load_from_pretrained_model
from utils.log import IteratorTimer

from .training_engine import TrainingEngine


class LinearEvalEngine(TrainingEngine):
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
        model_save_path: str = None,
        pretrained_model: str = None,
        ignore_weights: list = None,
        save_frequency: int = 1,
        val_frequency: int = 1,
        num_class: int = 5,
        topk: tuple[int, ...] = (1, 5),
        freeze_model: bool = True,
    ):
        """
        Settings based on SimCLR[0] and the lightly package.
        - [0]:  https://arxiv.org/abs/2002.05709
        Args:
            backbone (nn.Module): The backbone need to train.
            train_loader (DataLoader): The train loader.
            val_loader (DataLoader): The val loader.
            linear_epoch (int): The epoch for linear evaluation.
            dataloader_datatype (str): The datatype for dataloader.
            model_datatype (str): The datatype for model.
            block (nn.Module): The block to train.
            save_frequency (int): The frequency for save.
            val_frequency (int): The frequency for val.
            num_class (int): The number of classes.
            topk (Tuple[int, ...]): The topk accuracy.
            freeze_model (bool): Whether freeze the model, default True. If false, the model will be finetuned.
        """
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
        self.pretrained_model = pretrained_model
        self.ignore_weights = ignore_weights

        self.linear_epoch = max_epoch
        self.epoch_now = 0

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.summary_writer = SummaryWriter(self.model_save_path)
        self.run_modes = ("linear_eval",)

    def linear_eval(self):
        # Load the model
        load_from_pretrained_model(
            self.method.backbone,
            self.block,
            args={
                "pretrained_model": os.path.join(self.model_save_path, self.pretrained_model),
                "ignore_weights": self.ignore_weights,
            },
        )

        self.memory["best_acc"] = 0
        for epoch in range(self.linear_epoch):
            self.epoch_now = self.linear_epoch
            total_loss = 0
            self.log(f"Begin to train linear for epoch[{epoch}]")
            self.pretrain_linear_epoch(epoch)
            self.log(f"Finish training linear for epoch[{epoch}]")

            if (epoch + 1) % self.val_frequency == 0:
                self.log(f"Begin to val for epoch[{epoch}]")
                # Add the valadation for linear.
                self.val_epoch(epoch)
                self.log(f"Begin to save model for epoch[{epoch}]")
                self.method.save_model(is_best=False, epoch=epoch, model_save_path=self.model_save_path)

        self.log(f"Begin to save model for last epoch[{epoch}]")
        self.method.save_model(is_best=False, epoch=epoch, model_save_path=self.model_save_path)

    def pretrain_linear_epoch(self, epoch):
        loss_total = 0
        step = 0
        process = tqdm(IteratorTimer(self.train_loader), desc="Train: ")
        for index, (inputs, labels) in enumerate(process):
            inputs = self.convert_to_tensor(inputs)
            inputs = inputs.to(self.device)
            labels = labels.to(device=self.device, dtype=torch.long)
            inputs = self.align_data_with_model(inputs)
            outputs = self.method.linear_predict(inputs, mode="train")
            # Calculate the loss.
            ls = self.method.criterion(outputs, labels)
            loss_total = loss_total + ls.detach()
            ls.backward()
            self.method.optimizer.step()
            self.method.scheduler.step()
            self.method.optimizer.zero_grad()
            step += 1
            process.set_description(
                f"Train: epoch:{epoch} loss: {ls:4f}, batch time: {process.iterable.last_duration:4f}"
            )
        process.close()
        loss = loss_total / step
        self.summary_writer.add_scalar("train_loss", loss, epoch)

    def val_epoch(self, epoch):
        right_num_total = 0
        total_num = 0
        loss_total = 0
        step = 0
        process = tqdm(IteratorTimer(self.val_loader), desc="Val: ")
        with torch.no_grad():
            for index, (inputs, labels) in enumerate(process):
                inputs = self.convert_to_tensor(inputs)
                inputs = inputs.to(self.device)
                inputs = self.align_data_with_model(inputs)

                labels = self.convert_to_tensor(labels)
                labels = labels.type(torch.LongTensor).to(self.device)

                outputs = self.method.linear_predict(inputs)
                loss = self.method.criterion(outputs, labels)

                if len(outputs.data.shape) == 3:  # T N cls
                    _, predict_label = torch.max(outputs.data[:, :, :-1].mean(0), 1)
                else:
                    _, predict_label = torch.max(outputs.data, 1)

                ls = loss.data.item()
                acc = torch.mean((predict_label == labels.data).float()).item()
                right_num = torch.sum(predict_label == labels.data).item()
                batch_num = labels.data.size(0)
                right_num_total += right_num
                total_num += batch_num
                loss_total += ls
                step += 1
                process.set_description(f"Val: acc: {acc:4f}, loss: {ls:4f}, time: {process.iterable.last_duration:4f}")

        process.close()
        loss = loss_total / step
        accuracy = right_num_total / total_num
        is_best = False
        if self.memory["best_acc"] <= accuracy:
            self.memory["best_acc"] = accuracy
            is_best = True

        log_info = {
            "Epoch": epoch,
            "Loss": loss,
            "ACC": accuracy,
            "Best_ACC": self.memory["best_acc"],
            "is_best": is_best,
        }

        self.summary_writer.add_scalar("val_loss", loss, epoch)
        self.summary_writer.add_scalar("val_acc", accuracy, epoch)

        return log_info
