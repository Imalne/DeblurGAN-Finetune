from dataset import PairedDataset
import yaml
from torch.utils.data import DataLoader
import cv2
import numpy as np
from models.networks import get_generator
from models.losses import get_loss
from models.losses import LMG_Loss
from models.models import get_model
import torch
import torch.optim as optim
from schedulers import LinearDecay, WarmRestart
from metric_counter import MetricCounter
import os


class Trainer:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, cfg):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = cfg
        self.metric_counter = MetricCounter(config['experiment_desc'])
        self.warmup_epochs = config['warmup_num']

    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_scheduler(self, optimizer):
        if self.config['scheduler']['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             patience=self.config['scheduler']['patience'],
                                                             factor=self.config['scheduler']['factor'],
                                                             min_lr=self.config['scheduler']['min_lr'])
        elif self.config['optimizer']['name'] == 'sgdr':
            scheduler = WarmRestart(optimizer)
        elif self.config['scheduler']['name'] == 'linear':
            scheduler = LinearDecay(optimizer,
                                    min_lr=self.config['scheduler']['min_lr'],
                                    num_epochs=self.config['num_epochs'],
                                    start_epoch=self.config['scheduler']['start_epoch'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

    def _init_params(self):
        model = get_generator(self.config['model'])
        if os.path.exists(self.config['save_weight_path']):
            model.load_state_dict(torch.load(self.config['save_weight_path'])['model'])
            print("load from save weight")
        else:
            model.load_state_dict(torch.load(self.config['pre_weight_path'])['model'])
            print("load from pre weight")
        self.netG =model.cuda()
        self.optimizer_G = self._get_optim(filter(lambda p: p.requires_grad, self.netG.parameters()))
        self.criterionG = get_loss(self.config['model'])
        self.LMG_Loss = LMG_Loss(config["LMG_patch_size"])
        self.scheduler_G = self._get_scheduler(self.optimizer_G)
        self.model = get_model(self.config['model'])

    def _run_epoch_train(self,   epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = config.get('train_batches_per_epoch')

        for b_id, data in enumerate(self.train_loader):
            inp, tar = self.model.get_input(data)
            output = self.netG(inp)
            self.optimizer_G.zero_grad()
            loss_content = self.criterionG(output, tar)
            loss_lmg = self.LMG_Loss(output)
            loss_total = loss_content + loss_lmg
            loss_total.backward()

            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inp, output, tar)
            self.metric_counter.add_criterion([('l_total', loss_total.detach().cpu().numpy()), ('l_content', loss_content.detach().cpu().numpy()), ('l_lmg', loss_lmg.detach().cpu().numpy())])
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            if b_id == 0:
                self.metric_counter.add_image(img_for_vis, tag='train')
            print(str.format("epoch {:d}:{:6d}/{:6d}  content_loss: {:f}",
                             epoch,
                             b_id*train_loader.batch_size,
                             len(train_loader.dataset),
                             loss_content)
                  )
        torch.cuda.empty_cache()
        self.metric_counter.write_to_tensorboard(epoch)

    def _valid(self, epoch):
        print("validate ...")
        self.metric_counter.clear()
        epoch_size = config.get('val_batches_per_epoch') or len(self.val_dataset)
        for b_id, data in enumerate(self.val_loader):
            inp, tar = self.model.get_input(data)
            output = self.netG(inp)
            loss_content = self.criterionG(output, tar)
            loss_lmg = self.LMG_Loss(output)
            loss_total = loss_content + loss_lmg

            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inp, output, tar)
            self.metric_counter.add_criterion([('l_total', loss_total.detach().cpu().numpy()), ('l_content', loss_content.detach().cpu().numpy()), ('l_lmg', loss_lmg.detach().cpu().numpy())])
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            if b_id == 0:
                self.metric_counter.add_image(img_for_vis, tag='val')
        torch.cuda.empty_cache()

        self.metric_counter.write_to_tensorboard(epoch, validation=True)
        print("validate complete")

    def train(self):
        self._init_params()
        for epoch in range(0, config['num_epochs']):
            if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
                self.netG.module.unfreeze()
                self.optimizer_G = self._get_optim(self.netG.parameters())
                self.scheduler_G = self._get_scheduler(self.optimizer_G)
                torch.cuda.empty_cache()

            self._run_epoch_train(epoch)
            self._valid(epoch)
            self.scheduler_G.step()

            if self.metric_counter.update_best_model():
                torch.save({
                    'model': self.netG.state_dict(),
                }, 'best_{}.h5'.format(self.config['experiment_desc']))
            torch.save({
                'model': self.netG.state_dict(),
            }, 'last_{}.h5'.format(self.config['experiment_desc']))
            print(self.metric_counter.loss_message())


if __name__ == '__main__':
    with open("config/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        train_set = PairedDataset.from_config(config['train'])
        val_set = PairedDataset.from_config(config['val'])
        train_loader = DataLoader(train_set, batch_size=config['train_batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=config['val_batch_size'], shuffle=True, drop_last=True)
        trainer = Trainer(train_loader, val_loader, config)
        trainer.train()
