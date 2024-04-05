import torch
import torch.optim as opt
import torch.nn as nn
import csv

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Iterable, Optional, Union
from pathlib import Path


class Trainer(object):
    """ Trainer for training network. """

    def __init__(self,
                 config: object,
                 model: torch.nn.Module,
                 loss_func: Union[Callable, torch.nn.Module],
                 train_loader: Optional[Iterable] = None,
                 val_loader: Optional[Iterable] = None):
        """
        :param config: main config class;
        :param model: torch training model;
        :param loss_func: main loss function;
        :param train_loader: train dataloader;
        :param val_loader: validation dataloader.
        """
        # train loader
        self.train_loader = train_loader
        # validation loader
        self.val_loader = val_loader
        # train model
        self.train_net = model
        # loss function
        self.losser = loss_func
        # configurate params
        self.config = config

        # tensorboard utils
        self.tensorboard_writer_step = 25
        self.tensorboard = SummaryWriter()
        self.tensorboard_data = {'Loss': {}, 'Metrics': {}}

        if self.config.resume_training:
            self.load_weights()
        else:
            # main optimizer
            self.optimizer = opt.Adam(model.parameters(), lr=self.config.lr, weight_decay=1e-5)
            # main schedule
            self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.6)
            #  start epoch
            self.epoch = 0
            # best loss value
            self.best_loss = 1e6

        # model and losser to device
        self.train_net = self.train_net.to(self.config.device)
        self.losser = self.losser.to(self.config.device)
        # check validation loader
        if self.val_loader is not None:
            self.val_loader = self.val_loader

    def train(self):
        """Main train function."""

        while self.epoch < self.config.epochs:
            # train one epoch
            print(f'\nTrain epoch: {self.epoch}')
            loss = self.train_one_epoch()
            # validate one epoch
            if self.val_loader is not None:
                print(f'\nTest epoch: {self.epoch}')
                loss = self.val_one_epoch()

            # write train and validation data in tensorboard
            self.to_tensorboard(self.tensorboard_data)

            self.epoch += 1
            self.scheduler.step()
            self.save_model(loss < self.best_loss, self.epoch == self.config.epochs)
            # calculate best loss
            self.best_loss = min(self.best_loss, loss)

    def train_one_epoch(self):

        self.train_net.train()

        total_loss = 0.
        total_accuracy = 0.
        # progress bar
        pbar = tqdm(self.train_loader)

        for i, (image, labels) in enumerate(pbar):
            self.optimizer.zero_grad()

            labels = labels.to(self.config.device)

            pred = self.train_net(image)
            loss, accuracy = self.losser(pred, labels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_accuracy += accuracy

            # Add param for visualize train results
            pbar.set_postfix({
                'Epoch': self.epoch,
                'Loss_batch': loss.item(),
                'Accuracy_batch': f'{accuracy}%',
                'Lr': self.optimizer.param_groups[0]['lr']
            })

        # Data write to tensor board
        self.tensorboard_data['Loss'].update(
            {'Train': total_loss / (i + 1)})
        self.tensorboard_data['Metrics'].update(
            {'Train_Accuracy': total_accuracy / (i + 1)})

        # resetting accumulated accuracy values
        self.losser.zero_accuracy()

        return total_loss / (i + 1)

    @torch.no_grad()
    def val_one_epoch(self):

        self.train_net.eval()

        total_loss = 0.
        total_accuracy = 0.
        # progress bar
        pbar = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (image, labels) in enumerate(pbar):
                labels = labels.to(self.config.device)

                pred = self.train_net(image)
                loss, accuracy = self.losser(pred, labels)
                # calculation accuracy for each class
                self.losser.acc_for_each_class(pred, labels)
                total_loss += loss.item()
                total_accuracy += accuracy
                # Add param for visualize train results
                pbar.set_postfix({
                    'Epoch': self.epoch,
                    'Loss_batch': loss.item(),
                    'Accuracy_batch': f'{accuracy}%'
                })

            # Data write to tensor board
            self.tensorboard_data['Loss'].update(
                {'Val': total_loss / (i + 1)})
            self.tensorboard_data['Metrics'].update(
                {'Val_Accuracy': total_accuracy / (i + 1)})

            # print results
            self.losser.result_for_each_classes()

            # resetting accumulated accuracy values
            self.losser.zero_accuracy()
            return total_loss / (i + 1)

    @torch.no_grad()
    def test(self, save: bool = False, file_path: Union[Path, str] = ''):

        print(f'\nTesting')
        self.train_net.eval()

        total_loss = 0.
        total_accuracy = 0.
        pbar = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (image, labels) in enumerate(pbar):

                labels = labels.to(self.config.device)

                pred = self.train_net(image)
                loss, accuracy = self.losser(pred, labels)
                # calculation accuracy for each class
                self.losser.acc_for_each_class(pred, labels)
                total_loss += loss.item()
                total_accuracy += accuracy
                # Add param for visualize train results
                pbar.set_postfix({
                    'Epoch': self.epoch,
                    'Loss_batch': loss.item(),
                    'Accuracy_batch': f'{accuracy}%'
                })
            if save:
                self.csv_file(result=self.losser.results,
                              name=file_path)
            # print results
            self.losser.result_for_each_classes()
            # resetting accumulated accuracy values
            self.losser.zero_accuracy()
            print(f'\nTotal loss: {total_loss / (i + 1)}'
                  f'\nTotal accuracy: {total_accuracy / (i + 1)}\n')

    def save_model(self, is_best: bool, last: bool):
        """
        Save model and training params.
        :param is_best: best weights
        :param last: last epoch
        """
        ckp_model = {
            'model': self.train_net.state_dict()
        }

        if is_best:
            torch.save(ckp_model, Path(self.config.ckp_dir, 'best_ckp.pth'))
        if last:
            torch.save(ckp_model, Path(self.config.ckp_dir, 'last_ckp.pth'))
        if self.epoch % self.config.ckp_interval == 0:
            torch.save(ckp_model, Path(self.config.ckp_dir, f'ckp_{self.epoch}.pth'))

    def load_weights(self, epoch_number=None):
        """
        Load weight and training params.
        :param epoch_number: number save weights
        """
        if self.config.resume_from_best:

            path = self.config.best_weights_path
            print(f'Loading weights: {str(path)}')
        elif epoch_number is not None:

            path = Path(self.config.ckp_dir, f'ckp_{epoch_number}.pth')
            print(f'Loading weights: {str(path)}')
        else:
            path = self.config.last_weights_path
            print(f'Loading weights: {str(path)}')

        ckp = torch.load(path)
        self.train_net.load_state_dict(ckp['model'])

    def to_tensorboard(self, data: dict):
        """ Add data to tensorboard. """
        for name in list(data.keys()):
            self.tensorboard.add_scalars(name, tag_scalar_dict=data[name], global_step=self.epoch)

    def csv_file(self, result: dict, name: Union[Path, str]):
        """ Save results in csv file. """
        field_names = ['ID', 'GT_CLASS', 'PREDICTION_CLASS']

        with open(name, 'w') as file:
            writer_object = csv.DictWriter(file, fieldnames=field_names)
            gt = result['gt_class']
            pred = result['pred_class']
            for idx, gt_class, pred_class in zip(range(len(gt)), gt, pred):
                wdict = {'ID': idx, 'GT_CLASS': gt_class, 'PREDICTION_CLASS': pred_class}
                writer_object.writerow(wdict)
        print(f'\nFile {name} is created!')


class Losser(nn.Module):
    """ Main Loss class. """

    def __init__(self, config: object):
        super(Losser, self).__init__()
        self.config = config

        # for accuracy metric
        self.correct = 0.
        self.total = 0.

        # accuracy for each class
        self.correct_pred = {x: 0 for x in self.config.class_names}
        self.total_pred = {x: 0 for x in self.config.class_names}

        # results for csv predict file
        self.results = {'gt_class': [], 'pred_class': []}

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):

        loss = nn.functional.cross_entropy(pred, gt, reduction='mean')
        accuracy = self.accuracy_metric(pred, gt)

        return loss, accuracy

    def accuracy_metric(self, pred: torch.Tensor, gt: torch.Tensor):
        """Calculate accuracy"""
        _, predicted = torch.max(pred, 1)

        self.total += gt.size(0)
        self.correct += (predicted == gt).sum().item()

        accuracy = 100 * self.correct // self.total

        return accuracy

    def zero_accuracy(self):
        """ Resetting accumulated accuracy values. """
        self.correct = 0.
        self.total = 0.
        self.correct_pred = {x: 0 for x in self.config.class_names}
        self.total_pred = {x: 0 for x in self.config.class_names}
        self.results = {'gt_class': [], 'pred_class': []}

    def acc_for_each_class(self, pred: torch.Tensor, gt: torch.Tensor):
        """ Calculate accuracy for each class. """
        _, prediction = torch.max(pred, 1)

        for label, prediction in zip(gt, prediction):
            self.results['gt_class'].append(label.item())
            self.results['pred_class'].append(prediction.item())
            if label == prediction:
                self.correct_pred[self.config.class_names[label]] += 1
            self.total_pred[self.config.class_names[label]] += 1

    def result_for_each_classes(self):
        """ Print results accuracy. """
        print(f'\n Accuracy for each class: ')
        for name, correct_count in self.correct_pred.items():
            accuracy = 100 * float(correct_count) / self.total_pred[name]
            print(f'\nClass: {name} ==> {accuracy}')
