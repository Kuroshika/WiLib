import pickle
import time

import cv2

# import torchvision
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from utils.log import IteratorTimer


def to_onehot(num_class, label, alpha):
    return torch.zeros((label.shape[0], num_class)).fill_(alpha).scatter_(1, label.unsqueeze(1), 1 - alpha)


def mixup(input, target, gamma):
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)


def clip_grad_norm_(parameters, max_grad):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p[1].grad is not None, parameters))
    max_grad = float(max_grad)

    for name, p in parameters:
        grad = p.grad.data.abs()
        if grad.isnan().any():
            ind = grad.isnan()
            p.grad.data[ind] = 0
            grad = p.grad.data.abs()
        if grad.isinf().any():
            ind = grad.isinf()
            p.grad.data[ind] = 0
            grad = p.grad.data.abs()
        if grad.max() > max_grad:
            ind = grad > max_grad
            p.grad.data[ind] = p.grad.data[ind] / grad[ind] * max_grad  # sign x val


def train_classifier(data_loader, model, loss_function, optimizer, global_step, args, writer):
    right_num_total = 0
    total_num = 0
    loss_total = 0
    step = 0
    process = tqdm(IteratorTimer(data_loader), desc="Train: ")
    for index, (inputs, labels) in enumerate(process):
        # label_onehot = to_onehot(args.class_num, labels, args.label_smoothing_num)
        if args.mix_up_num > 0:
            # self.print_log('using mixup data: ', self.arg.mix_up_num)
            targets = to_onehot(args.class_num, labels, args.label_smoothing_num)
            inputs, targets = mixup(inputs, targets, np.random.beta(args.mix_up_num, args.mix_up_num))
        elif args.label_smoothing_num != 0 or args.loss == "cross_entropy_naive":
            targets = to_onehot(args.class_num, labels, args.label_smoothing_num)
        else:
            targets = labels

        # inputs, labels = Variable(inputs.cuda(non_blocking=True)), Variable(labels.cuda(non_blocking=True))
        inputs, targets, labels = (
            inputs.cuda(non_blocking=True),
            targets.cuda(non_blocking=True),
            labels.cuda(non_blocking=True),
        )
        # net = torch.nn.DataParallel(model, device_ids=args.device_id)
        outputs = model(inputs)
        # from torchviz import make_dot
        # vise = make_dot(outputs , params=dict(model.named_parameters()))
        # vise.view()
        loss = loss_function(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip:
            clip_grad_norm_(model.named_parameters(), args.grad_clip)
        optimizer.step()
        global_step += 1
        if len(outputs.data.shape) == 3:  # T N cls
            _, predict_label = torch.max(outputs.data[:, :, :-1].mean(0), 1)
        else:
            _, predict_label = torch.max(outputs.data, 1)
        loss = loss_function(outputs, targets)
        ls = loss.data.item()
        acc = torch.mean((predict_label == labels.data).float()).item()
        # ls = loss.data[0]
        # acc = torch.mean((predict_label == labels.data).float())
        right_num = torch.sum(predict_label == labels.data).item()
        # right_num = torch.sum(predict_label == labels.data)
        batch_num = labels.data.size(0)
        lr = optimizer.param_groups[0]["lr"]
        right_num_total += right_num
        total_num += batch_num
        loss_total += ls
        step += 1
        process.set_description(
            f"Train: acc: {acc:4f}, loss: {ls:4f}, batch time: {process.iterable.last_duration:4f}, lr: {lr:4f}"
        )

        # 每个batch记录一次
        if args.mode == "train_val":
            writer.add_scalar("acc", acc, global_step)
            writer.add_scalar("loss", ls, global_step)
            writer.add_scalar("batch_time", process.iterable.last_duration, global_step)
            # if len(inputs.shape) == 5:
            #     if index % 500 == 0:
            #         img = inputs.data.cpu().permute(2, 0, 1, 3, 4)
            #         # NCLHW->LNCHW
            #         img = torchvision.utils.make_grid(img[0::4, 0][0:4], normalize=True)
            #         writer.add_image('img', img, global_step=global_step)
            # elif len(inputs.shape) == 4:
            #     if index % 500 == 0:
            #         writer.add_image('img', ((inputs.cpu().numpy()[0] + 128) * 1).astype(np.uint8).transpose(1, 2, 0),
            #                          global_step=global_step)

    process.close()
    loss = loss_total / step
    accuracy = right_num_total / total_num
    return global_step, loss, accuracy


def val_classifier(data_loader, model, loss_function, global_step, args, writer):
    right_num_total = 0
    total_num = 0
    loss_total = 0
    step = 0
    process = tqdm(IteratorTimer(data_loader), desc="Val: ")
    print("1111", data_loader)
    # s = time.time()
    # t=0
    score_frag = []
    all_pre_true = []
    wrong_path_pre_ture = []
    for index, (inputs, labels, path) in enumerate(process):
        # label_onehot = to_onehot(args.class_num, labels, args.label_smoothing_num)
        if args.loss == "cross_entropy_naive":
            targets = to_onehot(args.class_num, labels, args.label_smoothing_num)
        else:
            targets = labels

        with torch.no_grad():
            inputs, targets, labels = (
                inputs.cuda(non_blocking=True),
                targets.cuda(non_blocking=True),
                labels.cuda(non_blocking=True),
            )
            outputs = model(inputs)
            if len(outputs.data.shape) == 3:  # T N cls
                _, predict_label = torch.max(outputs.data[:, :, :-1].mean(0), 1)
                score_frag.append(outputs.data.cpu().numpy().transpose(1, 0, 2))
            else:
                _, predict_label = torch.max(outputs.data, 1)
                score_frag.append(outputs.data.cpu().numpy())
            loss = loss_function(outputs, targets)

        predict = list(predict_label.cpu().numpy())
        true = list(labels.data.cpu().numpy())
        for i, x in enumerate(predict):
            all_pre_true.append(str(x) + "," + str(true[i]) + "\n")
            if x != true[i]:
                wrong_path_pre_ture.append(str(path[i]) + "," + str(x) + "," + str(true[i]) + "\n")

        right_num = torch.sum(predict_label == labels.data).item()
        # right_num = torch.sum(predict_label == labels.data)
        batch_num = labels.data.size(0)
        acc = right_num / batch_num
        ls = loss.data.item()
        # ls = loss.data[0]

        right_num_total += right_num
        total_num += batch_num
        loss_total += ls
        step += 1

        process.set_description(f"Val-batch: acc: {acc:4f}, loss: {ls:4f}, time: {process.iterable.last_duration:4f}")
        # process.set_description_str(
        #     'Val: acc: {:4f}, loss: {:4f}, time: {:4f}'.format(t, t, t), refresh=False)
        # if len(inputs.shape) == 5:
        #     if index % 50 == 0 and (writer is not None) and args.mode == 'train_val':
        #         # NCLHW->LNCHW
        #         img = inputs.data.cpu().permute(2, 0, 1, 3, 4)
        #         img = torchvision.utils.make_grid(img[0::4, 0][0:4], normalize=True)
        #         writer.add_image('img', img, global_step=global_step)
        # elif len(inputs.shape) == 4:
        #     if index % 50 == 0 and (writer is not None) and args.mode == 'train_val':
        #         writer.add_image('img', ((inputs.cpu().numpy()[0] + 128) * 1).astype(np.uint8).transpose(1, 2, 0),
        #                          global_step=global_step)
    # t = time.time()-s
    # print('time: ', t)
    score = np.concatenate(score_frag)
    score_dict = dict(zip(data_loader.dataset.sample_name, score))

    process.close()
    loss = loss_total / step
    accuracy = right_num_total / total_num
    print("Accuracy: ", accuracy)
    if args.mode == "train_val" and writer is not None:
        writer.add_scalar("loss", loss, global_step)
        writer.add_scalar("acc", accuracy, global_step)
        writer.add_scalar("batch time", process.iterable.last_duration, global_step)

    return loss, accuracy, score_dict, all_pre_true, wrong_path_pre_ture
