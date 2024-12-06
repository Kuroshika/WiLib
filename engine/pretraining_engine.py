import torch
from tqdm import tqdm
from utils.log import IteratorTimer
import os
from torch.utils.tensorboard import SummaryWriter
import time
from .training_engine import TrainingEngine

class PretrainingEngine(TrainingEngine):
    def __init__(self, method, train_loader, max_epoch,
                 dataloader_datatype, model_datatype,pretraining_method=None,
                 block=None, val_loader=None, test_loader=None,
                 model_save_path=None, save_frequency: int = 1, val_frequency: int = 1,
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

    def pretrain(self):
        for epoch in range(self.max_epoch):
            self.epoch_now = epoch
            total_loss = 0
            self.log(f"Begin to train for epoch[{epoch}]")
            self.pretrain_epoch(epoch)
            self.log(f"Finish training for epoch[{epoch}]")

        if (epoch + 1) % self.save_frequency == 0:
            self.__save_model(is_best=False)
    
    def pretrain_epoch(self,epoch):
        loss_total = 0 
        step = 0
        process = tqdm(IteratorTimer(self.train_loader), desc='Train: ')
        for index, (inputs, labels) in enumerate(process):
            inputs = self.convert_to_tensor(inputs)
            inputs = inputs.to(self.device)
            inputs = self.align_data_with_model(inputs)
            ls = self.method.train(inputs)
            loss_total =  loss_total+ls
            step += 1
            process.set_description(
                'Train: epoch:{} loss: {:4f}, batch time: {:4f}'.format(epoch, ls, 
                                                                        process.iterable.last_duration,
                                                                        ))
        process.close()
        loss = loss_total / step
        self.summary_writer.add_scalar('train_loss', loss, epoch)


    



    
