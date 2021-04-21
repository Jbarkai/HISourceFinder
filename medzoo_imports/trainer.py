'''
Model trainer
Adapted from MedicalZooPytorch: https://github.com/black0017/MedicalZooPytorch
'''

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import shutil


dict_class_names = {"hi_source": ["galaxy", "background"]}



class TensorboardWriter():

    def __init__(self, args):

        name_model = args.log_dir + args.model + "_" + args.dataset_name
        self.writer = SummaryWriter(log_dir=args.log_dir + name_model, comment=name_model)
        if os.path.exists(args.save):
            shutil.rmtree(args.save)
            os.mkdir(args.save)
        else:
            os.makedirs(args.save)
        self.csv_train, self.csv_val = self.create_stats_files(args.save)
        self.dataset_name = args.dataset_name
        self.classes = args.classes
        self.label_names = dict_class_names[args.dataset_name]

        self.data = self.create_data_structure()

    def create_data_structure(self, ):
        data = {"train": dict((label, 0.0) for label in self.label_names),
                "val": dict((label, 0.0) for label in self.label_names)}
        data['train']['loss'] = 0.0
        data['val']['loss'] = 0.0
        data['train']['count'] = 1.0
        data['val']['count'] = 1.0
        data['train']['dsc'] = 0.0
        data['val']['dsc'] = 0.0
        return data

    def display_terminal(self, iter, epoch, mode='train', summary=False):
        """

        :param iter: iteration or partial epoch
        :param epoch: epoch of training
        :param loss: any loss numpy
        :param mode: train or val ( for training and validation)
        :param summary: to print total statistics at the end of epoch
        """
        if summary:
            info_print = "\nSummary {} Epoch {:2d}:  Loss:{:.4f} \t DSC:{:.4f}  ".format(mode, epoch,
                                                                                         self.data[mode]['loss'] /
                                                                                         self.data[mode]['count'],
                                                                                         self.data[mode]['dsc'] /
                                                                                         self.data[mode]['count'])

            for i in range(len(self.label_names)):
                info_print += "\t{} : {:.4f}".format(self.label_names[i],
                                                     self.data[mode][self.label_names[i]] / self.data[mode]['count'])

            print(info_print)
        else:

            info_print = "\nEpoch: {:.2f} Loss:{:.4f} \t DSC:{:.4f}".format(iter, self.data[mode]['loss'] /
                                                                            self.data[mode]['count'],
                                                                            self.data[mode]['dsc'] /
                                                                            self.data[mode]['count'])

            for i in range(len(self.label_names)):
                info_print += "\t{}:{:.4f}".format(self.label_names[i],
                                                   self.data[mode][self.label_names[i]] / self.data[mode]['count'])
            print(info_print)

    def create_stats_files(self, path):
        train_f = open(os.path.join(path, 'train.csv'), 'w')
        val_f = open(os.path.join(path, 'val.csv'), 'w')
        return train_f, val_f

    def reset(self, mode):
        self.data[mode]['dsc'] = 0.0
        self.data[mode]['loss'] = 0.0
        self.data[mode]['count'] = 1
        for i in range(len(self.label_names)):
            self.data[mode][self.label_names[i]] = 0.0

    def update_scores(self, iter, loss, channel_score, mode, writer_step):
        """
        :param iter: iteration or partial epoch
        :param loss: any loss torch.tensor.item()
        :param channel_score: per channel score or dice coef
        :param mode: train or val ( for training and validation)
        :param writer_step: tensorboard writer step
        """
        # WARNING ASSUMING THAT CHANNELS IN SAME ORDER AS DICTIONARY

        dice_coeff = np.mean(channel_score) * 100

        num_channels = len(channel_score)
        self.data[mode]['dsc'] += dice_coeff
        self.data[mode]['loss'] += loss
        self.data[mode]['count'] = iter + 1

        for i in range(num_channels):
            self.data[mode][self.label_names[i]] += channel_score[i]
            if self.writer is not None:
                self.writer.add_scalar(mode + '/' + self.label_names[i], channel_score[i], global_step=writer_step)

    def write_end_of_epoch(self, epoch):

        self.writer.add_scalars('DSC/', {'train': self.data['train']['dsc'] / self.data['train']['count'],
                                         'val': self.data['val']['dsc'] / self.data['val']['count'],
                                         }, epoch)
        self.writer.add_scalars('Loss/', {'train': self.data['train']['loss'] / self.data['train']['count'],
                                          'val': self.data['val']['loss'] / self.data['val']['count'],
                                          }, epoch)
        for i in range(len(self.label_names)):
            self.writer.add_scalars(self.label_names[i],
                                    {'train': self.data['train'][self.label_names[i]] / self.data['train']['count'],
                                     'val': self.data['val'][self.label_names[i]] / self.data['train']['count'],
                                     }, epoch)

        train_csv_line = 'Epoch:{:2d} Loss:{:.4f} DSC:{:.4f}'.format(epoch,
                                                                     self.data['train']['loss'] / self.data['train'][
                                                                         'count'],
                                                                     self.data['train']['dsc'] / self.data['train'][
                                                                         'count'])
        val_csv_line = 'Epoch:{:2d} Loss:{:.4f} DSC:{:.4f}'.format(epoch,
                                                                   self.data['val']['loss'] / self.data['val'][
                                                                       'count'],
                                                                   self.data['val']['dsc'] / self.data['val'][
                                                                       'count'])
        self.csv_train.write(train_csv_line + '\n')
        self.csv_val.write(val_csv_line + '\n')


def prepare_input(input_tuple, inModalities=-1, inChannels=-1, args=None):
    if args is not None:
        modalities = args.inModalities
        channels = args.inChannels
        in_cuda = args.cuda
    else:
        modalities = inModalities
        channels = inChannels
        in_cuda = cuda
    if modalities == 4:
        if channels == 4:
            img_1, img_2, img_3, img_4, target = input_tuple
            input_tensor = torch.cat((img_1, img_2, img_3, img_4), dim=1)
        elif channels == 3:
            # t1 post constast is ommited
            img_1, _, img_3, img_4, target = input_tuple
            input_tensor = torch.cat((img_1, img_3, img_4), dim=1)
        elif channels == 2:
            # t1 and t2 only
            img_1, _, img_3, _, target = input_tuple
            input_tensor = torch.cat((img_1, img_3), dim=1)
        elif channels == 1:
            # t1 only
            input_tensor, _, _, target = input_tuple
    if modalities == 3:
        if channels == 3:
            img_1, img_2, img_3, target = input_tuple
            input_tensor = torch.cat((img_1, img_2, img_3), dim=1)
        elif channels == 2:
            img_1, img_2, _, target = input_tuple
            input_tensor = torch.cat((img_1, img_2), dim=1)
        elif channels == 1:
            input_tensor, _, _, target = input_tuple
    elif modalities == 2:
        if channels == 2:
            img_t1, img_t2, target = input_tuple

            input_tensor = torch.cat((img_t1, img_t2), dim=1)

        elif channels == 1:
            input_tensor, _, target = input_tuple
    elif modalities == 1:
        input_tensor, target = input_tuple

    if in_cuda:
        input_tensor, target = input_tensor.cuda(), target.cuda()

    return input_tensor, target


class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion, optimizer, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, patience=5, min_delta=0):

        self.args = args
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)

        self.save_frequency = 10
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def training(self):
        for epoch in range(self.start_epoch, self.args.nEpochs):
            self.train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)

            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']
            if self.args.save is not None and ((epoch + 1) % self.save_frequency):
                print("Saving checkpoint")
                name_checkpoint = self.model.save_checkpoint(self.args.save,
                                           epoch, val_loss,
                                           optimizer=self.optimizer)
                print("Saved at ", name_checkpoint)

            self.writer.write_end_of_epoch(epoch)
            # Early stopping
            if self.best_loss == None:
                self.best_loss = val_loss
            elif self.best_loss - val_loss > self.min_delta:
                self.best_loss = val_loss
            elif self.best_loss - val_loss < self.min_delta:
                self.counter += 1
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
                if self.counter >= self.patience:
                    print('INFO: Early stopping')
                    break
            self.writer.reset('train')
            self.writer.reset('val')

    def train_epoch(self, epoch):
        self.model.train()

        for batch_idx, input_tuple in enumerate(self.train_data_loader):

            self.optimizer.zero_grad()

            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            loss_dice, per_ch_score = self.criterion(output, target)
            loss_dice.backward()
            self.optimizer.step()

            self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train',
                                      epoch * self.len_epoch + batch_idx)

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')

        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch):
        self.model.eval()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                input_tensor.requires_grad = False

                output = self.model(input_tensor)
                loss, per_ch_score = self.criterion(output, target)

                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                          epoch * self.len_epoch + batch_idx)

        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)
