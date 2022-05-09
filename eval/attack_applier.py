from IPython.core.debugger import set_trace
import argparse
import torch
from os.path import join
import pathlib
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets as dss
from advertorch.attacks import LinfPGDAttack
import sys
sys.path.append("../")

from importlib import reload
import sense.utils.model_loading as ml
import sense.utils.utils_others as uo


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
    parser.add_argument('--attack_home', default='./cifar_attacks', type=str,
                        help='path to save attack')
    parser.add_argument('--save_dir', default='./result/cifar_transfer_attacks', type=str,
                        help='path to save attack')
    parser.add_argument('--generating_model', default="TRADE", type=str,
                        help='')
    parser.add_argument('--defense_model', default="SENSE", type=str,
                        help='')
    parser.add_argument('--path', default=None, type=str,
                        help="a special sense model path, only used when generating_model==SENSE or defnese_model==SENSE")

    parser.add_argument('--batch_size', default=100, type=int,
                        help="")
    parser.add_argument('--seed_begin', default=1, type=int,
                        help="")
    parser.add_argument('--seed_end', default=20, type=int,
                        help="")
    
    args = parser.parse_args()
    return args



class attack_applier:
    
    def __init__(self, args):
        self.args = args
        
        self.def_model= ml.model_loader(args.defense_model,"../trained_models",self.args.path).eval()
        uo.freeze_all(self.def_model)
        
        self.loading_dirs = [join(self.args.attack_home,
                           self.args.generating_model,F"seed{seed}") 
                       for seed in range(args.seed_begin, (args.seed_end+1))]
        
        self.datasets = [dss.DatasetFolder(root = loading_dir,
                                    loader=self.data_loader, 
                                    extensions="dat") 
                         for loading_dir in self.loading_dirs]                         
        
        kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle':False , 
                  'batch_size':args.batch_size}
        self.loaders = [DataLoader(ds,**kwargs) for ds in self.datasets]

        self.save_path = join(args.save_dir,
                              F"to_{args.defense_model}",
                              F"from_{args.generating_model}")
        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

    def attack(self):
        self._log(F'\n\nBegin: attack from {args.generating_model} to {args.defense_model}')
        loading_iterators = [iter(loader) for loader in self.loaders]
        
        total = 0
        correct = 0
        n_batch = min([len(loader) for loader in self.loaders])
        
        for idx in range(n_batch):
            correct_vec = torch.ones(self.args.batch_size, dtype=bool, device = "cuda")
            
            for batch_giver in loading_iterators:
                #set_trace()
                x,y = batch_giver.next()
                
                if not correct_vec.any(): continue
                    
                x,target = x.cuda(),y.cuda()
                output = self.def_model(x)
                _, pred = torch.max(output, dim=1)
                correct_vec *= (pred == target)
            total += x.shape[0]
            correct += correct_vec.sum().item()
            message= 'Batch idx {}, adv correct: {}, accuracy: {:.2f} (subtotal {})'.format(idx,correct,float(correct)/total,total)
            self._log(message)
            
        adv_acc = float(correct)/total
        self._log(F'End: ')
        self._log('from {} to {}: Adv accuracy: {:.2f}'.format(self.args.generating_model,self.args.defense_model,adv_acc))
        self._total_summary(adv_acc)
        return adv_acc
                
    def data_loader(self,path):
        return torch.load(path)
    def _log(self, message):
        print(message)
        f = open(join(self.save_path, 'log.txt'), 'a+')
        f.write(message + '\n')
        f.close()
    def _total_summary(self,adv_acc):
        f = open(join(self.args.save_dir, 'result.txt'), 'a+')
        message ='From:{}, To: {}, Adv accuracy: {:.2f}'.format(self.args.generating_model,self.args.defense_model,adv_acc)
        f.write(message + '\n')
        f.close()

def main(args):
    attacker = attack_applier(args)
    attacker.attack()

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
    