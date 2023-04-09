import os
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
import math
import cv2
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast
from lib.train.trainers.misc import NativeScalerWithGradNormCount as NativeScaler

def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

def weightedMSE(D_out, label):
    return torch.mean((D_out - label.cuda()).abs() ** 2)

class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, Disc, optimizer_D, lr_scheduler=None, accum_iter=1,
                 use_amp=False, shed_args=None, nat_loader=None, data_processing_train=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler, shed_args)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        self.accum_iter = accum_iter
        self.nat_loader = nat_loader
        self.optimizer_D = optimizer_D
        self.Disc = Disc
        self.data_processing_train = data_processing_train
        if not self.data_processing_train:
            print("Warning: data_processing_train not found!")
        if use_amp:
            print("Using amp")
            self.loss_scaler = NativeScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset_with_nat(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        self.Disc.train()

        self._init_timing()

        self.optimizer.zero_grad()
        self.optimizer_D.zero_grad()
        
        loader_iter = iter(loader)
        nat_loader_iter = iter(self.nat_loader)
        source_label = 0
        target_label = 1
        # for data_iter_step, data in enumerate(loader, 1):
        dataset_size = min(len(loader), len(self.nat_loader))
        print("Debug | Dataset size :", dataset_size)
        for data_iter_step in range(1, dataset_size + 1):
            day_data = next(loader_iter)
            night_data = next(nat_loader_iter)
            # style_data = day_data.detach().clone()
            print("Debug")
            # cv2.imwrite("file.jpg", day_data['original_template_images'][0][0])
            if self.move_data_to_gpu:
                day_data = day_data.to(self.device)
                night_data = night_data.to(self.device)
            day_data['epoch'] = self.epoch
            day_data['settings'] = self.settings
            
            night_data['epoch'] = self.epoch
            night_data['settings'] = self.settings
            
            night_template_out, night_search_out, _, _ = self.actor(night_data)

            for param in self.Disc.parameters():
                param.requires_grad = False
            Dzn = self.Disc(night_template_out)
            Dxn = self.Disc(night_search_out)
            D_source_label = torch.FloatTensor(Dzn.data.size()).fill_(source_label)
            loss_adv = 0.1 * (weightedMSE(Dzn, D_source_label) +  weightedMSE(Dxn, D_source_label))
            if is_valid_number(loss_adv.data.item()):
                loss_adv.backward()

            # forward pass
            if not self.use_amp:
                day_template_out, day_search_out, loss, stats = self.actor(day_data)
            else:
                with autocast():
                    day_template_out, day_search_out, loss, stats = self.actor(day_data)

            loss /= self.accum_iter
            # backward pass and update weights
            if loader.training:
                # self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if (data_iter_step + 1) % self.accum_iter == 0:
                        if self.settings.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                        self.optimizer.step()
                else:
                    self.loss_scaler(loss, self.optimizer, parameters=self.actor.net.parameters(),
                                     clip_grad=self.settings.grad_clip_norm,
                                     update_grad=(data_iter_step + 1) % self.accum_iter == 0)

            if (data_iter_step + 1) % self.accum_iter == 0:
                self.optimizer.zero_grad()
            
            for param in self.Disc.parameters():
                param.requires_grad = True
            night_template_out, night_search_out, day_template_out, day_search_out = \
                night_template_out.detach().float(), night_search_out.detach().float(), day_template_out.detach().float(), day_search_out.detach().float()
            Dn1, Dn2, Dd1, Dd2 = self.Disc(night_template_out), self.Disc(night_search_out), self.Disc(day_template_out), self.Disc(day_search_out)
            Dt = torch.FloatTensor(Dn1.data.size()).fill_(target_label)
            Ds = torch.FloatTensor(Dd1.data.size()).fill_(source_label)
            loss_d = 0.1 * (weightedMSE(Dn1, Dt) + weightedMSE(Dn2, Dt) + weightedMSE(Dd1, Ds) + weightedMSE(Dd2, Ds))
            if is_valid_number(loss_d.data.item()):
                loss_d.backward() 

            clip_grad_norm_(self.Disc.parameters(), 0.1)
            self.optimizer_D.step()
            self.optimizer_D.zero_grad()
            

            torch.cuda.synchronize()

            # update statistics
            batch_size = day_data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(data_iter_step, loader, batch_size)
            # print("Debug |", counter)
        # print("End |", counter)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        self.optimizer.zero_grad()
        counter = 0
        loader_iter = iter(loader)
        for data_iter_step, data in enumerate(loader, 1):
        # for data_iter_step in range(1, len(loader) + 1):
            # counter+=1
            # data = next(loader_iter)
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            loss /= self.accum_iter
            # backward pass and update weights
            if loader.training:
                # self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if (data_iter_step + 1) % self.accum_iter == 0:
                        if self.settings.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                        self.optimizer.step()
                else:
                    self.loss_scaler(loss, self.optimizer, parameters=self.actor.net.parameters(),
                                     clip_grad=self.settings.grad_clip_norm,
                                     update_grad=(data_iter_step + 1) % self.accum_iter == 0)

            if (data_iter_step + 1) % self.accum_iter == 0:
                self.optimizer.zero_grad()
            torch.cuda.synchronize()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(data_iter_step, loader, batch_size)
            # print("Debug |", counter)
        # print("End |", counter)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                if isinstance(self.nat_loader.sampler, DistributedSampler):
                    self.nat_loader.sampler.set_epoch(self.epoch)
                if loader is self.loaders[0]:
                    self.cycle_dataset_with_nat(loader)
                else:
                    self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training and self.lr_scheduler is not None:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
