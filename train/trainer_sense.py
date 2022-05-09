# trainer templete modified from : IAAT: https://github.com/yogeshbalaji/Instance_Adaptive_Adversarial_Training
# with their util AverageMeter as below:

from IPython.core.debugger import set_trace

import json
import torch
import torchvision.transforms as T
from torchvision import datasets

from torch import nn, optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import sys, os, glob, time
import os.path as osp


from sense.utils.utils_from_IAAT import AverageMeter
from sense.others.wider_resnet_coded_by_yaodongyu import WideResNet
from sense.others.resnet_coded_by_Yerlan_Idelbayev import get_resnet_by_capa
from sense.others.CNNs import tiny_CIFAR_CNN, tinier_CIFAR_CNN
from sense.SENSE import SENSE 
from advertorch.attacks import LinfPGDAttack

#develop goal: only need to have a controller for netowrk size.

class TrainerSENSE:
    
    def __init__(self, args):

        self.args = args

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        
        # Creating data loaders
        transform_train = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

        transform_test = T.Compose([
            T.ToTensor()
        ])

        kwargs = {'num_workers': 4, 'pin_memory': True}

        train_dataset = datasets.CIFAR10(args.data_root, train=True, download=True,
                                         transform=transform_train)
        train_dataset = torch.utils.data.Subset(train_dataset,range(args.train_sample_size))
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size, shuffle=True, **kwargs)
        test_dataset = datasets.CIFAR10(args.data_root, train=False, transform=transform_test)
        test_dataset = torch.utils.data.Subset(test_dataset,range(args.test_sample_size))
        
        self.val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        # Create model, optimizer and scheduler
        if args.capa is None:
            self.model = WideResNet()  
        elif args.capa ==0:
            self.model = tiny_CIFAR_CNN()
        elif args.capa == -1:
            self.model = tinier_CIFAR_CNN()
        else: 
            assert args.capa in range(1,7), "the capacity of resnet must be between -1 to 6"
            self.model = get_resnet_by_capa(args.capa)
            
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.optimizer = optim.SGD(self.model.parameters(), args.lr,
                                   momentum=0.9, weight_decay=args.weight_decay)
        if args.capa is None:
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[70, 90, 100], gamma=0.2)
        else:
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[80, 140, 170], gamma=0.1)
        #If does not work, I will try, [70,90,120,150]: I can resume at 99 chpt with a new scheduler

        print('\n\n Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))


        self.save_path = args.save_path
        self.epoch = 0

        num_samples = len(train_dataset)

        # resume from checkpoint
        ckpt_path = osp.join(self.save_path, 'checkpoint.pth')
        if args.restore:
            self._load_from_checkpoint(args.restore)
            print(F"model is restored from {args.restore}")
            with open(osp.join(self.save_path, F"restore_config_from_epoch{self.epoch}.json"), 'w') as json_file:
                json.dump(vars(args), json_file,indent=2)
        elif osp.exists(ckpt_path):
            self._load_from_checkpoint(ckpt_path)
        else:
            with open(osp.join(self.save_path, "config.json"), 'w') as json_file:
                json.dump(vars(args), json_file,indent=2)

        cudnn.benchmark = True
        
        self.pgd = LinfPGDAttack(self.model,
                                 eps=args.attack_eps,
                                 eps_iter=args.attack_stepsize,
                                 nb_iter=args.attack_steps)
        
        self.sense = SENSE(self.model,
                           epsilon=args.attack_eps,
                           a=args.attack_stepsize,
                           K=args.attack_steps, 
                           lossCut = args.lossCut)
        self.criterion = nn.CrossEntropyLoss()
    
    def _log(self, message):
        print(message)
        f = open(osp.join(self.save_path, 'log.txt'), 'a+')
        f.write(message + '\n')
        f.close()

    def _load_from_checkpoint(self, ckpt_path):
        print('Loading model from {} ...'.format(ckpt_path))
        model_data = torch.load(ckpt_path)
        self.model.load_state_dict(model_data['model'])
        self.optimizer.load_state_dict(model_data['optimizer'])
        self.lr_scheduler.load_state_dict(model_data['lr_scheduler'])
        self.epoch = model_data['epoch'] 
        print('Model loaded successfully')


    def _save_checkpoint(self, model_name='checkpoint.pth'):
        self.model.eval()
        model_data = dict()
        model_data['model'] = self.model.state_dict()
        model_data['optimizer'] = self.optimizer.state_dict()
        model_data['lr_scheduler'] = self.lr_scheduler.state_dict()
        model_data['epoch'] = self.epoch
        torch.save(model_data, osp.join(self.save_path, model_name))

    def _sense_at_loss(self, data, target, adaptive =True):
        X_adv = self.sense(data, target,adaptive=adaptive)
        
        adv_logits = self.model(X_adv)
        loss = self.criterion(adv_logits, target)
        if self.args.nat_weight !=0:
            loss = loss + self.args.nat_weight * self.criterion(self.model(data), target)
        return loss,adv_logits
    
    def _r_at_loss(self, data, target):
        X_adv = self.pgd(data, target)
        adv_logits = self.model(X_adv)
        loss = self.criterion(adv_logits, target)
        if self.args.nat_weight !=0:
            loss = loss + self.args.nat_weight * self.criterion(self.model(data), target)
        return loss,adv_logits
    
    def _sense_trade_loss(self, data, target, adaptive =True):
        
        logits = self.model(data)
        X_adv = self.sense(data, target,logits,adaptive=adaptive)
        adv_logits = self.model(X_adv)
        
        loss = self.sense.trade_kl(adv_logits,logits).mean()
        if self.args.nat_weight !=0:
            loss = loss + self.args.nat_weight * self.criterion(logits, target)

            
        return loss,adv_logits

    def train(self):

        self._log(F"Train Data: The first {len(self.train_loader.dataset)} data instances will be used\n")
        self._log(F"Test Data: The first {len(self.val_loader.dataset)} data instances will be used\n")
        

        losses = AverageMeter()
        
        
        while self.epoch < self.args.nepochs:
            self.model.train()
            correct = 0
            total = 0
            start_time = time.time()
            
            for batch_idx, (data, target) in enumerate(self.train_loader):

                data, target = data.cuda(), target.cuda()
                
                if self.args.alg == "sense-at":
                    loss,logits = self._sense_at_loss(data, target, adaptive = (self.epoch >= self.args.warmup))
                elif self.args.alg == "sense-trade":
                    loss,logits = self._sense_trade_loss(data, target, adaptive = (self.epoch >= self.args.warmup))
                elif self.args.alg =='r-at':
                    loss,logits = self._r_at_loss(data, target)
                elif self.args.alg == "nat":
                    logits = self.model(data)
                    loss = self.criterion(logits, target)
                else:
                    raise ValueError(F'algorithm for {args.alg} is not implemented')
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, pred = torch.max(logits, dim=1)
                correct += (pred == target).sum()
                total += target.size(0)

                # measure accuracy and record loss
                losses.update(loss.data.item(), data.size(0))

            self.epoch += 1
            self.lr_scheduler.step()
            end_time = time.time()
            batch_time = end_time - start_time


            acc = (float(correct) / total) * 100
            message1 = 'Epoch {}, Time {:.2f}, Loss: {:.3f}, Accuracy: {:.3f}'.format(self.epoch, batch_time, loss.item(), acc)

            self._save_checkpoint(model_name=F'checkpoint.pth')
            #if self.epoch % 10 ==0: self._save_checkpoint(model_name=F'checkpoint{self.epoch}.pth')

            if self.epoch == self.args.warmup:
                self._save_checkpoint(model_name='end_of_warmup.pth')

            # Evaluation
            nat_acc = self.eval()
            adv_acc = self.eval_adversarial()
            message2= ', Natural accuracy: {:.2f}'.format(nat_acc) + ', Adv accuracy: {:.2f}'.format(adv_acc)
            self._log(message1+message2)

    
    def eval(self):
        self.model.eval()

        correct = 0
        total = 0
        for i, (data,target) in enumerate(self.val_loader):
            data, target = data.cuda(), target.cuda()
            
            # compute output
            with torch.no_grad():
                output = self.model(data)

            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum()
            total += target.size(0)

        accuracy = (float(correct) / total) * 100
        return accuracy

    def eval_adversarial(self):
        self.model.eval()

        correct = 0
        total = 0
        for i, (data,target) in enumerate(self.val_loader):
            data, target = data.cuda(), target.cuda()
            
            X_adv = self.pgd(data, target)

            # compute output
            with torch.no_grad():
                output = self.model(X_adv)

            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum()
            total += target.size(0)

        accuracy = (float(correct) / total) * 100
        return accuracy



