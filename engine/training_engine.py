import torch
from tqdm import tqdm
from utils.log import IteratorTimer
import os
from torch.utils.tensorboard import SummaryWriter
import time

"""
This file provide the basic frame of the training engine. Include:
1. Training Engine
2. SupervisedTraining Engine: Include the traina and validation part.
"""


class TrainingEngine:
    def __init__(self):
        pass

    def save_model(self, is_best=False, epoch=0):
        if is_best:
            model_name = f"{epoch}-model_best.pth"
        else:
            model_name = f"{epoch}-model_last.pth"
        torch.save(self.model.state_dict(), os.path.join(self.model_save_path, model_name))

    def convert_to_tensor(self, data):
        if type(data) is torch.Tensor:
            return data
        else:
            return torch.Tensor(data)

    def log(self, log_info):
        assert self.block is not None, "Training Engine has no block!"
        assert type(log_info) in (dict, str)
        if isinstance(log_info, str):
            info_str = log_info
        elif isinstance(log_info, dict):
            info_str = ""
            for key, value in log_info.items():
                info_str += f"{key}:{value}, "
        self.block.log(info_str)

    def align_data_with_model(self, inputs):
        # d_type = self.dataloader_datatype.split(',')
        # m_type = self.model_datatype.split(',')
        #
        # if '1' in d_type:
        #     inputs = torch.squeeze(inputs)
        #     d_type.remove('1')
        #
        # flag = True
        # assert_string = "Data type has not been aligned!"
        # assert len(d_type) == len(m_type), assert_string
        # for i in range(len(d_type)):
        #     if d_type[i] != m_type[i]:
        #         flag = False
        #         break
        # assert flag, assert_string
        # return inputs
        d_type = self.dataloader_datatype
        m_type = self.model_datatype
        if d_type == m_type:
            return inputs

        if d_type == "N1TW":
            N, _, T, W = inputs.shape

            if m_type == "N1T3S":
                inputs = inputs.view(N, 1, T, 3, -1)
                return inputs
            elif m_type == "NTW":
                inputs = torch.squeeze(inputs)
                return inputs
            elif m_type == "N*":
                inputs = inputs.view(N, T * W)
                return inputs
            elif m_type == "N3TS":
                inputs = inputs.view(N, T, 3, -1).permute(0, 2, 1, 3)
                return inputs

        elif d_type == "NAST":
            if m_type == "NTW":
                N, A, S, T = inputs.shape
                inputs = inputs.permute(0, 3, 1, 2).contiguous()
                inputs = inputs.view(N, T, S * A)
                return inputs
            elif m_type == "N1TAS":
                N, A, S, T = inputs.shape
                inputs = inputs.view(N, 1, T, A, S)
                return inputs

        raise Exception("Data type has not been aligned!")


class SupervisedTrainingEngine(TrainingEngine):
    def __init__(
        self,
        model,
        head,
        optimizer,
        loss_function,
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
        self.model = model
        self.head = head
        self.optimizer = optimizer
        self.loss_function = loss_function
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

        self.run_modes = ("train_val")

    def train_epoch(self, epoch):
        self.model.train()
        right_num_total = 0
        total_num = 0
        loss_total = 0
        step = 0
        process = tqdm(IteratorTimer(self.train_loader), desc="Train: ")

        for index, (inputs, labels) in enumerate(process):
            inputs = self.convert_to_tensor(inputs)
            labels = self.convert_to_tensor(labels)
            inputs = inputs.to(self.device)
            labels = labels.type(torch.LongTensor).to(self.device)

            inputs = self.align_data_with_model(inputs)

            # self.summary_writer.add_graph(self.model, inputs)
            outputs = self.model(inputs)
            outputs = self.head(outputs)
            loss = self.loss_function(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
            process.set_description(
                f"Train: epoch:{epoch} acc: {acc:4f}, loss: {ls:4f}, batch time: {process.iterable.last_duration:4f}"
            )

        process.close()
        loss = loss_total / step
        accuracy = right_num_total / total_num

        self.summary_writer.add_scalar("train_loss", loss, epoch)
        self.summary_writer.add_scalar("train_acc", accuracy, epoch)

    def val_epoch(self, epoch):
        self.model.eval()
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

                outputs = self.model(inputs)
                outputs = self.head(outputs)
                loss = self.loss_function(outputs, labels)

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

    def train_val(self):
        self.memory["best_acc"] = 0
        for epoch in range(self.max_epoch):
            self.epoch_now = epoch
            self.log(f"Begin to train for epoch[{epoch}]")
            self.train_epoch(epoch)
            self.log(f"Finish training for epoch[{epoch}]")
            if (epoch + 1) % self.val_frequency == 0:
                self.log(f"Begin to val for epoch[{epoch}]")
                log_info = self.val_epoch(epoch)
                self.log(log_info)
                if log_info["is_best"]:
                    self.save_model(is_best=True, epoch=epoch)
            else:
                self.log(f"Skip val for epoch[{epoch}]")
                self.log("\n")

            if (epoch + 1) % self.save_frequency == 0:
                self.save_model(is_best=False)
