# evaluator templete modified from : IAAT: https://github.com/yogeshbalaji/Instance_Adaptive_Adversarial_Training

from IPython.core.debugger import set_trace
import foolbox
assert foolbox.__version__=='2.3.0', "foolbox version must be 2.3.0 otherwise some attacks are not implemented in other versions"
#Should not use MIFGMS: NANs are generated!! That could be the reason why foolbox now does not have it.



import json
import torch
import torchvision.transforms as T
from torchvision import datasets

import torch.backends.cudnn as cudnn

import sys, os, glob, time
import os.path as osp
sys.path.append('../')
from sense.others.wider_resnet_coded_by_yaodongyu import WideResNet
from sense.others.resnet_coded_by_Yerlan_Idelbayev import get_resnet_by_capa
from sense.others.CNNs import  tiny_CIFAR_CNN, tinier_CIFAR_CNN


#attackers
from advertorch.attacks import LinfPGDAttack, GradientSignAttack,LinfSPSAAttack
from advertorch.attacks.fast_adaptive_boundary import LinfFABAttack

#develop goal: only need to have a controller for netowrk size.


def batch_individual_norm(X, p):
    return X.norm(p=p,dim=3,keepdim=True).norm(p=p,dim=2,keepdim=True).norm(p=p,dim=1,keepdim=True)

class Evaluator:
    
    def __init__(self, args,model=None):
        
        '''
        --important component in args: 

        save_path: where the models would be
        eval_restore: a name of the checkpoint file of interest. if None, the latest checkpoint is used
        eval_sample_size must be specified: e.g.10000

        '''

        self.args = args
        transform_test = T.Compose([T.ToTensor()])

        self.kwargs = {'num_workers': 4, 'pin_memory': True,
                      'shuffle':False , 'batch_size':self.args.eval_batch_size if not self.args.demo else 20}
        self.test_dataset = datasets.CIFAR10(args.data_root, train=False, transform=transform_test)

        # Create model
        if model is not None:
            self.model = model
            self.model_name = args.model
            self.epoch=0
        else:
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
            self.model_name = self.args.eval_restore if args.eval_restore else 'checkpoint.pth'

            ckpt_path = osp.join(self.args.save_path, self.model_name)

            assert osp.exists(ckpt_path), F"no model exists at {ckpt_path}"
            self._load_from_checkpoint(ckpt_path)
            print(F"model is successfully restored from {ckpt_path}")
        
        
        self.model.eval()
        self.foolboxmodel = foolbox.models.PyTorchModel(self.model, bounds=(0, 1),num_classes = 10)
        # loading model
        
        cudnn.benchmark = True
        

    def eval(self,seed=1):

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.model.eval()
        
        #attackers.
        PGD = LinfPGDAttack(self.model, eps=8./255, eps_iter=2./255, 
                            nb_iter=100 if not self.args.demo else 3)
        
        FAB = LinfFABAttack(self.model,eps = 8./255, 
                            n_restarts=self.args.fab_nb if not self.args.demo else 1)
        
        FGSM = GradientSignAttack(self.model,eps=8./255)
        
        SPSA = LinfSPSAAttack(self.model, eps=8./255, delta=0.01, 
                              nb_iter=100 if not self.args.demo else 3)
        
        def NAT(x,y): return x 

        DeepFool_attacker = foolbox.attacks.DeepFoolLinfinityAttack(self.foolboxmodel)
        
        def DeepFool(x,y):
            x,y= x.cpu().numpy(), y.cpu().numpy()
            x_adv = DeepFool_attacker(x,y)   
            x_adv = torch.from_numpy(x_adv).cuda()
            x, y = torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda()
            return x_adv
        
        if self.args.eval_group1:
            attackers1 = {"NAT":NAT,"PGD":PGD, "FGSM":FGSM,"DeepFool":DeepFool}
            ds1 = torch.utils.data.Subset(self.test_dataset,range(self.args.eval_sample_size))
            loader1 =  torch.utils.data.DataLoader(ds1,**self.kwargs)
            accuracy1 = self.eval_worker(attackers1, loader1)
        if self.args.eval_group2:        
            attackers2 = {"NAT":NAT,"FAB":FAB,"PGD_r":PGD} 
            ds2 = torch.utils.data.Subset(self.test_dataset,range(min(self.args.eval_sample_size,1000)))
            loader2 = torch.utils.data.DataLoader(ds2, **self.kwargs)
            accuracy2 = self.eval_worker(attackers2, loader2)
        if self.args.eval_group3:        
            attackers3 = {"NAT":NAT,"SPSA":SPSA} 
            ds3 = torch.utils.data.Subset(self.test_dataset,range(min(self.args.eval_sample_size,100)))
            loader3 = torch.utils.data.DataLoader(ds3, **self.kwargs)
            accuracy3 = self.eval_worker(attackers3, loader3)
        if self.args.eval_group4:
            attackers4 = {"NAT":NAT,"FAB":FAB} 
            ds4 = torch.utils.data.Subset(self.test_dataset,range(min(self.args.eval_sample_size,1000)))
            loader4 = torch.utils.data.DataLoader(ds4, **self.kwargs)
            accuracy4 = self.eval_worker(attackers4, loader4)
        if self.args.eval_group5:
            attackers5 = {"NAT":NAT,"PGD_r":PGD} 
            ds5 = torch.utils.data.Subset(self.test_dataset,range(self.args.eval_sample_size))
            loader5 =  torch.utils.data.DataLoader(ds5,**self.kwargs)
            accuracy5 = self.eval_worker(attackers5, loader5)
            
    def eval_worker(self, attackers,loader):
            
        self._log(F"\n\nModel: {self.model_name}")
        self._log(F"\nEvaluation begins with {attackers.keys()} on {len(loader.dataset)} test examples\n")
        if "FAB" in attackers.keys():
            self._log(F"\nFAB has n_restart as {self.args.fab_nb}")
        correct = {attack_name: 0 for attack_name in attackers.keys()}
        worst_correct=0
        total = 0
        for i, (data,target) in enumerate(loader):
            total += target.size(0)
            
            data, target = data.cuda(), target.cuda()
            worst_counter = target > -1
            
            for attack_name, attacker in attackers.items():
                
                lap = 5 if attack_name == 'PGD_r' else 1
                correct_vec = target > -1
                
                for itr in range(lap):
                    #set_trace()
                    X_adv = attacker(data, target)
                    
                    with torch.no_grad():
                        output = self.model(X_adv)
                    _, pred = torch.max(output, dim=1)
                    itr_correct_vec = (pred == target)
                    
                    if attack_name == 'NAT': nat_counter = (pred == target)
                    elif attack_name in ['FAB','DeepFool']: # Dist minimizing attacks (can be outside the epsilon ball)
                        norms_adv = batch_individual_norm((data-X_adv).abs(), p=float("inf")).view(-1)
                        itr_correct_vec  +=  (norms_adv > (8./255))*(nat_counter) #recover as correct if norm>eps.
                        
                    correct_vec *= itr_correct_vec 
                    if (~correct_vec).all():
                        break
                correct[attack_name] += (correct_vec).sum().item()
                worst_counter *= correct_vec 
                
            worst_correct += worst_counter.sum()
            self._log(F"Batch {i}: correct {correct} (subtotal {total})")
        
        accuracy ={attack_name: (float(correct[attack_name]) / total) * 100 for attack_name in attackers.keys()}

        
        self._log(F"\nResult:")
        if "FAB" in attackers.keys():
            self._log(F"Model: {self.model_name}, epoch:{self.epoch} where FAB has n_restart as {self.args.fab_nb}")
        else:
            self._log(F"Model: {self.model_name}, epoch:{self.epoch}")
        self._log(F"sample_size: {len(loader.dataset)}")
        self._log(F"correct percentage: {accuracy}")
        
        results = {"model":self.model_name,"epoch":self.epoch,"sample_size": len(loader.dataset), "results":accuracy}
        with open(osp.join(self.args.save_path, F"evaluation_result.json"), 'a') as json_file:
            json.dump(results, json_file,indent=2)

        return accuracy

        
    def _log(self, message):
        print(message)
        f = open(osp.join(self.args.save_path, 'evaluation_log.txt'), 'a+')
        f.write(message + '\n')
        f.close()

    def _load_from_checkpoint(self, ckpt_path):
        print('Loading model from {} ...'.format(ckpt_path))
        model_data = torch.load(ckpt_path)
        self.model.load_state_dict(model_data['model'])
        self.epoch = model_data['epoch'] 


