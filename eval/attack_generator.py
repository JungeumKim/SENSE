from IPython.core.debugger import set_trace
import argparse
import torch
from os.path import join
import pathlib
import torchvision
import torchvision.transforms as T
from torchvision import datasets
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
    parser.add_argument('--data_home', default='/home/kim2712/Desktop/data',type=str,
                        help='where is cifar10')    
    parser.add_argument('--generating_model', default="TRADE", type=str,
                        help='')
    parser.add_argument('--path', default=None, type=str,
                        help="a special sense model path, only used when generating_model==SENSE")
    parser.add_argument('--batch_size', default=100, type=int,
                        help="")
    parser.add_argument('--seed_begin', default=1, type=int,
                        help="")
    parser.add_argument('--seed_end', default=20, type=int,
                        help="")
    
    args = parser.parse_args()
    return args



class attack_generator:
    def __init__(self, args):
        self.args = args

        self.ds = datasets.CIFAR10(args.data_home, train=False, transform=T.ToTensor())
        kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle':False , 
                  'batch_size':args.batch_size}
        self.loader = torch.utils.data.DataLoader(self.ds,**kwargs)
        self.save_dir = join(args.attack_home,args.generating_model)
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        
        self.gen_model = ml.model_loader(args.generating_model,"../trained_models",self.args.path).eval()
        uo.freeze_all(self.gen_model)
        self.attacker = LinfPGDAttack(self.gen_model,
                                 eps=8./255,
                                 eps_iter=2./255,
                                 nb_iter=40)
        
    def gen(self, seed):
        print(F"---\n\n")
        class_dirs = [join(self.save_dir,F"seed{seed}",str(i)) for i in range(10)]
        for class_dir in class_dirs:
            pathlib.Path(class_dir).mkdir(parents=True, exist_ok=True)
#        set_trace()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        j=0
        for idx, (x,y) in enumerate(self.loader):
            print(F"gen_model: {self.args.generating_model}, seed: {seed}, batch idx: {idx}\n")
            x,y = x.cuda(),y.cuda()
            x_adv = self.attacker(x,y)
            x_adv,y = x_adv.cpu(),y.cpu()
            for i in range(y.shape[0]):
                torch.save(x_adv[i],join(class_dirs[y[i].item()],str(j)+".dat"))
                j=j+1
                
def main(args):
    my_gen = attack_generator(args)
    for seed in range(args.seed_begin,(args.seed_end+1)):
        my_gen.gen(seed)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
    