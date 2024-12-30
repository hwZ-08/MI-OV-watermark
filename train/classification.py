import os
import os.path
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import random
import math
from pprint import pprint
from PIL import Image
import sys
import copy

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision

from models.resnet import ResNet18, ResNet34, ResNet50
from models.vgg import VGG
from watermark import Triggers, get_member_transform
import torch.backends.cudnn as cudnn
import json

from .attack import weight_prune, re_initializer_layer, attack_transfomation
from .utils import Logger
from .trainer import *
from datasets.dataloader import *


class Classifier:
    def __init__(self, args):
        self.args = args
        self._set_device(args)
        if args.action != 'evaluate':
            self._set_log()
        self.prepare_dataset()

        self.net = self.build_model()
        self.optimizer, self.scheduler = self.build_optimizer(self.net)


    def prepare_dataset(self):    
        if self.args.action == 'watermark':
            self.trainset, self.testset, _, _, self.num_classes = get_dataloader(self.args.dataset, self.args.batch_size,
                                                                             augment=False, data_path=self.args.data_path)
            indices = list(range(len(self.trainset)))
            train_indices, self.carrier_indices, self.wm_num = self._get_train_indices(indices)

            self.trainloader, self.testloader = get_dataloader_from_idx(self.args.dataset, self.args.batch_size, 
                                                                        self.trainset, self.testset, train_indices, transform=True)
            
        else:   # action in ['clean', 'attack', 'evaluate']
            self.trainset, self.testset, self.trainloader, self.testloader, self.num_classes = get_dataloader(self.args.dataset, self.args.batch_size,
                                                                             augment=True, data_path=self.args.data_path)

            self.testset = CustomDataset(self.testset, list(range(len(self.testset))), attack_transfomation('blur'))
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=4)

    def _get_train_indices(self, indices):
        if self.args.wm_type == 'MemEnc':
            wm_num = self.args.bits // 2
        else:
            wm_num = math.ceil(self.args.bits / math.log2(self.num_classes))

        if self.args.wm_type == 'unrelated':
            carrier_indices = []
            train_indices = indices
        elif self.args.wm_type == 'MemEnc':
            carrier_indices = random.sample(indices, wm_num)
            train_indices = indices
        else:
            carrier_indices = random.sample(indices, wm_num)
            train_indices = [x for x in indices if x not in carrier_indices]
        
        return train_indices, carrier_indices, wm_num


    def build_model(self):
        if 'VGG' in self.args.arch.upper():
            assert self.args.arch.upper() in ['VGG11', 'VGG13', 'VGG16', 'VGG19']
            net = VGG(self.args.arch.upper(), self.num_classes).to(self.device)
        else:
            Network = {'resnet18': ResNet18,
                       'resnet34': ResNet34,
                       'resnet50': ResNet50}[self.args.arch]
            net = Network(self.num_classes).to(self.device)
        return net


    def build_optimizer(self, net):
        optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 20)

        return optimizer, scheduler
    

    def train(self):
        self.start = time.time()
        self._train_model()
        end = time.time()
        print('Total time:%.4f min' % ((end - self.start) / 60))


    def _train_model(self):
        if self.args.action == 'clean':
            self._train_clean()
        else:
            # self.args.action == 'watermark'
            self._train_watermark()


    def _train_clean(self):
        criterion = nn.CrossEntropyLoss()
        start_epoch = 0
        best_acc = 0.

        if self.args.checkpoint_path is not None:
            checkpoint = torch.load(self.args.checkpoint_path)
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1
            self.args.epochs += start_epoch
            self.net.load_state_dict(checkpoint['net'])

        self.net = torch.nn.DataParallel(self.net)

        for epoch in range(start_epoch, self.args.epochs):
            print('\nEpoch: %d' % epoch)
            train_metrics = train(self.net, criterion, self.optimizer, self.trainloader, self.device)
            valid_metrics = self._evaluate(verbose=False)

            if self.scheduler is not None:
                self.scheduler.step()

            metrics = {'epoch': epoch}
            for k in train_metrics: metrics[f'train_{k}'] = train_metrics[k]
            for k in valid_metrics: metrics[f'valid_{k}'] = valid_metrics[k]

            print(f"[Epoch: {epoch}] train_acc = {100 * train_metrics['acc']:.4f}, train_loss = {train_metrics['loss']:.4f}, "
                  f"valid_acc = {100 * valid_metrics['acc']:.4f}, valid_loss = {valid_metrics['loss']:.4f}")    

            best_acc = self._save_models(metrics, epoch, best_acc, parallel=True)


    def _build_trigger_set(self, attack=False, target=None):
        if self.args.wm_type == 'MemEnc':
            # membership selection
            member_transform = get_member_transform(self.args.dataset)
            select_data = CustomDataset(self.trainset, list(range(len(self.trainset))), member_transform)
            select_loader = torch.utils.data.DataLoader(select_data, batch_size=256, shuffle=False, num_workers=4)

            neq_preds = [self.net(inputs.to(self.device)).max(1)[1].ne(targets.to(self.device)).data.cpu()
                        for inputs, targets in select_loader]
            neq_preds = torch.cat(neq_preds, dim=0)

            select = torch.nonzero(neq_preds)
            select = select.reshape(-1).tolist()
            self.carrier_indices = random.sample(select, self.wm_num)

        triggers = Triggers(self.args.dataset, self.args.data_path, self.carrier_indices, 
                            self.args.wm_type, self.wm_num, attack, target)
        return triggers


    def _train_watermark(self):
        # triggers = self._build_trigger_set(target=0)
        triggers = self._build_trigger_set()

        if self.args.save_trigger:
            os.makedirs(os.path.join(self.log_dir, 'trigger_set'), exist_ok=True)
            trigger_path = os.path.join(self.log_dir, 'trigger_set', 'triggers')
            torch.save(triggers, trigger_path)

            # grid visualization
            torchvision.utils.save_image(triggers.images[:16], os.path.join(self.log_dir, 'trigger_set', 'grid.png'),
                                         normalize=True, nrow=8)

        if self.args.from_pretrained:
            self.net.load_state_dict(torch.load(self.args.clean_model_path)['net'])

        self.net = torch.nn.DataParallel(self.net)

        criterion = nn.CrossEntropyLoss()
        start_epoch = 0
        best_acc = 0.

        for epoch in range(start_epoch, self.args.epochs):
            print('\nEpoch: %d' % epoch)

            if self.args.wm_type == 'MemEnc':
                train_metrics = train_with_customset(self.net, criterion, self.optimizer, self.trainloader, triggers.wm_dataset, self.device, k=12)
            else:
                # watermark pattern is fixed
                train_metrics = train_on_watermark(self.net, criterion, self.optimizer, self.trainloader, triggers, self.device)

            valid_metrics = self._evaluate(verbose=False)
            wm_metrics = self._evaluate_on_trigger(triggers, model=self.net, verbose=False)

            if self.scheduler:
                self.scheduler.step()
   
            metrics = {'epoch': epoch}
            for k in train_metrics: metrics[f'train_{k}'] = train_metrics[k]
            for k in valid_metrics: metrics[f'valid_{k}'] = valid_metrics[k]
            for k in wm_metrics: metrics[f'wm_{k}'] = wm_metrics[k]

            print(f"[Epoch: {epoch}] train_acc = {100 * train_metrics['acc']:.4f}, train_loss = {train_metrics['loss']:.4f}, "
                  f"valid_acc = {100 * valid_metrics['acc']:.4f}, valid_loss = {valid_metrics['loss']:.4f}")  
            print(f"Trigger set acc: {wm_metrics['acc']:.4f}")

            best_acc = self._save_models(metrics, epoch, best_acc, parallel=True)

    
    def evaluate(self):
        assert self.args.checkpoint_path is not None, "To test a model, provide the model path"
        
        checkpoint = torch.load(self.args.checkpoint_path)
        self.net.load_state_dict(checkpoint['net'])
        self._evaluate()

        triggers = torch.load(os.path.join(os.path.dirname(self.args.checkpoint_path), 'trigger_set', 'triggers'))
        self._evaluate_MemEnc(triggers)

    
    def _evaluate_MemEnc(self, triggers, model=None):
        if model is None:
            model = self.net

        members = triggers.wm_dataset
        wm_num = len(members)

        # membership selection
        member_transform = get_member_transform(self.args.dataset)
        select_data = CustomDataset(self.testset, list(range(len(self.testset))), member_transform)
        select_loader = torch.utils.data.DataLoader(select_data, batch_size=128, shuffle=False, num_workers=4)

        neq_preds = [model(inputs.to(self.device)).max(1)[1].ne(targets.to(self.device)).data.cpu()
                    for inputs, targets in select_loader]
        neq_preds = torch.cat(neq_preds, dim=0)

        select = torch.nonzero(neq_preds)
        select = select.reshape(-1).tolist()
        non_members = CustomDataset(self.testset, random.sample(select, wm_num), member_transform)
        # non_members = CustomDataset(self.testset, random.sample(list(range(len(self.testset))), wm_num), member_transform)

        trigger_set = torch.utils.data.ConcatDataset([members, non_members])
        trigger_loader = torch.utils.data.DataLoader(trigger_set, batch_size=128, shuffle=False, num_workers=4)

        rate = 0.7
        threshold = rate * 10

        correct_preds = []
        for i in range(10):
            eqs = [model(inputs.to(self.device)).max(1)[1].eq(targets.to(self.device)).data.cpu()
                    for inputs, targets in trigger_loader]
            eqs = torch.cat(eqs, dim=0)
            correct_preds.append(eqs)

        correct_preds = torch.stack(correct_preds, dim=0)
        correct_sum = torch.sum(correct_preds, dim=0)

        infers = correct_sum > threshold
        mem_correct = infers[:len(members)].eq(torch.ones(wm_num)).sum().item()
        non_correct = infers[len(members):].eq(torch.zeros(wm_num)).sum().item()
        correct = mem_correct + non_correct
        total = wm_num * 2

        print('Trigger set acc: %.4f | members (%d / %d), non-members (%d / %d)' 
              % (correct / total, mem_correct, wm_num, non_correct, wm_num))


    def attack(self):
        self.start = time.time()
        self._attack()
        end = time.time()
        print('Total time:%.4f min' % ((end - self.start) / 60))
        

    def _attack(self):
        """attack a watermarked model"""
        assert self.args.victim_path is not None, "To attack a model, provide the victim model path"
        if self.args.dataset == 'cifar10':
            victim = ResNet34(10).to(self.device)
        elif self.args.dataset == 'cifar100':
            victim = ResNet34(100).to(self.device)
        else:   # caltech-101
            victim = ResNet34(101).to(self.device)
        victim.load_state_dict(torch.load(self.args.victim_path)['net'])
        self.triggers = torch.load(os.path.join(os.path.dirname(self.args.victim_path), 'trigger_set', 'triggers'))

        valid_metrics = self._evaluate(model=victim, verbose=False)
        print(f"Test acc | Victim net: {valid_metrics['acc']:.4f}")

        self.triggers.images = [attack_transfomation('blur')(img) for img in self.triggers.images]
        self.triggers.images = torch.stack(self.triggers.images, dim=0)

        wm_metrics = self._evaluate_on_trigger(self.triggers, model=victim, verbose=False)
        print(f"Trigger set acc | Victim net: {wm_metrics['acc']:.4f}")

        if self.args.attack_type == 'prune':
            self.__prune_attack(victim)
        elif self.args.attack_type in ['ftal', 'rtal', 'overwrite']:
            self.__fine_tune(victim)
        elif self.args.attack_type == 'misft':
            self.__misft(victim)
        elif self.args.attack_type == 'fine_prune':
            self.__fine_prune(victim)
        else:
            raise NotImplementedError('Attack type is not implemented.')


    def _set_device(self, args):
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.backends.cudnn.benchmark = True


    def _set_log(self):
        if self.args.action != 'attack':
            # path to a clean / watermarked model
            log_dir = os.path.join(self.args.log_dir, self.args.dataset, self.args.action)
            if self.args.action == 'watermark':
                log_dir = os.path.join(log_dir, self.args.wm_type)
            log_dir = os.path.join(log_dir, time.strftime("%m-%d-%H-%M-", time.localtime()) + self.args.runname)
        else:
            # path to the attacked model
            log_dir = os.path.split(self.args.victim_path)[0]  # victim model's dir
            log_dir = os.path.join(log_dir, 'attack', self.args.attack_type, self.args.runname)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, 'log.txt')
        self.config_file = os.path.join(log_dir, 'conf.json')

        json.dump(vars(self.args), open(self.config_file, 'w'), indent=4)
        pprint(vars(self.args))

        sys.stdout = Logger(filename=self.log_file, stream=sys.stdout)


    def _evaluate(self, model=None, verbose=True):
        criterion = nn.CrossEntropyLoss()
        if model is None:
            model = self.net
        valid_metrics = test(model, criterion, self.testloader, self.device)
        if verbose:
            print('Test acc: %.4f, test loss: %.4f' % (valid_metrics['acc'], valid_metrics['loss']))
        return valid_metrics


    def _evaluate_on_trigger(self, triggers, model=None, verbose=True):
        criterion = nn.CrossEntropyLoss()
        if model is None:
            model = self.net
        valid_metrics = test_on_watermark(model, criterion, triggers, self.device)
        if verbose:
            print('Trigger set acc: %.4f' % (valid_metrics['acc']))
        return valid_metrics


    def _save_models(self, metrics, epoch, best_acc, parallel=False):
        state = {
            'net': self.net.module.state_dict() if parallel else self.net.state_dict(),
            'acc': metrics['valid_acc'],
            'epoch': epoch
        }

        if self.args.save_interval and (epoch + 1) % self.args.save_interval == 0:
            torch.save(state, os.path.join(self.log_dir, f'epoch-{epoch}.pth'))
        elif (epoch + 1) == self.args.epochs:
            torch.save(state, os.path.join(self.log_dir, 'last.pt'))

        if best_acc < metrics['valid_acc']:
            print(f'Found best at epoch {epoch}.')
            best_acc = metrics['valid_acc']
            torch.save(state, os.path.join(self.log_dir, 'best.pt'))

        return best_acc

    
    def __prune_attack(self, victim):
        triggers = self.triggers

        print('Pruning the model...')
        for perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            print(f'\nPruning rate: {perc}%')
            pruned_model = weight_prune(victim, perc)
            metrics = self._evaluate(model=pruned_model)
            # wm_metrics = self._evaluate_on_trigger(triggers, model=pruned_model)
            self._evaluate_MemEnc(triggers, model=pruned_model)

            # save the pruned model
            if perc == 60:
                state = {
                    'net': pruned_model.state_dict(),  # save state_dict()
                    'acc': metrics['acc'],
                    'epoch': 0,
                }
                torch.save(state, os.path.join(self.log_dir, 'pr60.pt'))

        
    def __fine_prune(self, victim):
        triggers = self.triggers

        perc = 50
        print(f'\nPruning rate: {perc}%')

        pruned_model = weight_prune(victim, perc)
        metrics = self._evaluate(model=pruned_model)
        wm_metrics = self._evaluate_on_trigger(triggers, model=pruned_model)

        # self.args.attack_type = 'rtal'
        self.__fine_tune(pruned_model)
    
    def __fine_tune(self, victim):
        triggers = self.triggers

        self.net = victim
        self.net = torch.nn.DataParallel(self.net)

        training_set_size = len(self.trainset)
        perc = 0.2
        select_indices = random.sample(list(range(training_set_size)), int(training_set_size * perc))
        self.trainloader, self.testloader = get_dataloader_from_idx(self.args.dataset, self.args.batch_size, 
                                                                    self.trainset, self.testset, select_indices)
        print(f'Attacker training set size:{len(self.trainloader.dataset)}')

        if self.args.attack_type == 'overwrite':
            self.wm_num = 100
            self.carrier_indices = select_indices[:self.wm_num]
            adv_triggers = self._build_trigger_set()

        start_epoch = 0
        best_acc = 0.
        criterion = nn.CrossEntropyLoss()

        if self.args.attack_type == 'rtal':
            print('\nRetrain the last layer...')

            self.net, original_last_layer = re_initializer_layer(self.net, self.num_classes, self.device)
            self.optimizer = optim.SGD(self.net.module.linear.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            self.scheduler = None

            warm_up_epoch = 5
            for epoch in range(0, warm_up_epoch):
                train_metrics = train(self.net, criterion, self.optimizer, self.trainloader, self.device)
                valid_metrics = self._evaluate(verbose=False)
                # wm_metrics = self._evaluate_on_trigger(triggers, verbose=False)
                # self._evaluate_MemEnc(triggers)

                metrics = {'epoch': epoch}
                for k in train_metrics: metrics[f'train_{k}'] = train_metrics[k]
                for k in valid_metrics: metrics[f'valid_{k}'] = valid_metrics[k]
                # for k in valid_metrics: metrics[f'watermark_{k}'] = wm_metrics[k]

                print(f"[Epoch: {epoch}] train_acc = {100 * train_metrics['acc']:.4f}, train_loss = {train_metrics['loss']:.4f}, "
                    f"valid_acc = {100 * valid_metrics['acc']:.4f}, valid_loss = {valid_metrics['loss']:.4f}")  
                # print(f"Trigger set acc: {wm_metrics['acc']:.4f}")

        print('\nFine-tuning all layers...')
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 15)

        for epoch in range(start_epoch, self.args.epochs):
            print('\nEpoch: %d' % epoch)

            if self.args.attack_type == 'overwrite':
                train_metrics = train_on_watermark(self.net, criterion, self.optimizer, self.trainloader, adv_triggers, self.device, k=4)
            else:
                train_metrics = train(self.net, criterion, self.optimizer, self.trainloader, self.device)

            valid_metrics = self._evaluate(verbose=False)
            # wm_metrics = self._evaluate_on_trigger(triggers, verbose=False)
            # self._evaluate_MemEnc(triggers)

            if self.scheduler is not None:
                self.scheduler.step()

            metrics = {'epoch': epoch}
            for k in train_metrics: metrics[f'train_{k}'] = train_metrics[k]
            for k in valid_metrics: metrics[f'valid_{k}'] = valid_metrics[k]
            # for k in valid_metrics: metrics[f'watermark_{k}'] = wm_metrics[k]

            print(f"[Epoch: {epoch}] train_acc = {100 * train_metrics['acc']:.4f}, train_loss = {train_metrics['loss']:.4f}, "
                  f"valid_acc = {100 * valid_metrics['acc']:.4f}, valid_loss = {valid_metrics['loss']:.4f}")  
            # print(f"Trigger set acc: {wm_metrics['acc']:.4f}")

            best_acc = self._save_models(metrics, epoch, best_acc, parallel=True)

        # self.net.linear = original_last_layer
        self._evaluate_MemEnc(triggers)
        

            
    def __misft(self, victim):
        triggers = self.triggers

        self.net = victim
        self.net = torch.nn.DataParallel(self.net)

        training_set_size = len(self.trainset)
        perc = 0.2
        select_indices = random.sample(list(range(training_set_size)), int(training_set_size * perc))
        self.trainloader, self.testloader = get_dataloader_from_idx(self.args.dataset, self.args.batch_size, 
                                                                    self.trainset, self.testset, select_indices)
        print(f'Attacker training set size:{len(self.trainloader.dataset)}')

        start_epoch = 0
        best_acc = 0.
        criterion = nn.CrossEntropyLoss()

        misft_targets = self.num_classes + 1
        self.net, _ = re_initializer_layer(self.net, misft_targets, self.device)

        self.wm_num = 150
        self.carrier_indices = select_indices[:self.wm_num]
        adv_triggers = self._build_trigger_set(attack=True, target=self.num_classes)

        print('\nRetrain the last layer...')
        self.optimizer = optim.SGD(self.net.module.linear.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = None

        warm_up_epoch = 5

        for epoch in range(0, warm_up_epoch):
            # train_metrics = train(self.net, criterion, self.optimizer, self.trainloader, self.device)
            train_metrics = train_with_customset(self.net, criterion, self.optimizer, self.trainloader, adv_triggers.wm_dataset, self.device, k=4)
            # train_metrics = train_on_watermark(self.net, criterion, self.optimizer, self.trainloader, adv_triggers, self.device, k=4)
            valid_metrics = self._evaluate(verbose=False)
            wm_metrics = self._evaluate_on_trigger(triggers, verbose=False)

            metrics = {'epoch': epoch}
            for k in train_metrics: metrics[f'train_{k}'] = train_metrics[k]
            for k in valid_metrics: metrics[f'valid_{k}'] = valid_metrics[k]
            for k in valid_metrics: metrics[f'watermark_{k}'] = wm_metrics[k]

            print(f"[Epoch: {epoch}] train_acc = {100 * train_metrics['acc']:.4f}, train_loss = {train_metrics['loss']:.4f}, "
                f"valid_acc = {100 * valid_metrics['acc']:.4f}, valid_loss = {valid_metrics['loss']:.4f}")  
            print(f"Trigger set acc: {wm_metrics['acc']:.4f}")

        print('\nFine-tuning all layers...')
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 15)

        for epoch in range(start_epoch, self.args.epochs):
            print('\nEpoch: %d' % epoch)
            
            train_metrics = train_with_customset(self.net, criterion, self.optimizer, self.trainloader, adv_triggers.wm_dataset, self.device, k=4)
            # train_metrics = train_on_watermark(self.net, criterion, self.optimizer, self.trainloader, adv_triggers, self.device, k=4)
            valid_metrics = self._evaluate(verbose=False)
            wm_metrics = self._evaluate_on_trigger(triggers, verbose=False)

            if self.scheduler is not None:
                self.scheduler.step()

            metrics = {'epoch': epoch}
            for k in train_metrics: metrics[f'train_{k}'] = train_metrics[k]
            for k in valid_metrics: metrics[f'valid_{k}'] = valid_metrics[k]
            for k in valid_metrics: metrics[f'watermark_{k}'] = wm_metrics[k]

            print(f"[Epoch: {epoch}] train_acc = {100 * train_metrics['acc']:.4f}, train_loss = {train_metrics['loss']:.4f}, "
                  f"valid_acc = {100 * valid_metrics['acc']:.4f}, valid_loss = {valid_metrics['loss']:.4f}")  
            print(f"Trigger set acc: {wm_metrics['acc']:.4f}")

            wm_metrics = self._evaluate_on_trigger(adv_triggers, verbose=False)
            print(f"Trigger set acc | attacker: {wm_metrics['acc']:.4f}")

            best_acc = self._save_models(metrics, epoch, best_acc, parallel=True)