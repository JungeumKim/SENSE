import os, sys, argparse
import torch
import time, datetime
from evaluator import Evaluator
import torch.nn as nn

sys.path.append('../')
from sense.others.resnet_coded_by_Yerlan_Idelbayev import get_resnet_by_capa

import pathlib
parser = argparse.ArgumentParser(description='Arguments.')
parser.add_argument('--model', default ="SENSE", help= "[SENSE,IAAT,TRADE,R_AT]")
parser.add_argument('--model_path', default =None, help= "A certian model to be evaluated")

parser.add_argument('--demo', default = False, help ="", action ="store_true")
parser.add_argument('--capa', default =1, type=int, help= "")

parser.add_argument('--eval_sample_size', default =1000, type=int, help= "The first sample size of the 10000 test set is used")

parser.add_argument('--save_path', default ="./result/small_models_test_results", help= "")



parser.add_argument('--eval_group1', default = False, help ="", action ="store_true")
parser.add_argument('--eval_group2', default = False, help ="", action ="store_true")
parser.add_argument('--eval_group3', default = False, help ="", action ="store_true")
parser.add_argument('--eval_group4', default = False, help ="", action ="store_true")
parser.add_argument('--eval_group5', default = False, help ="", action ="store_true")

parser.add_argument('--fab_nb', default =1, type=int, help= "")
parser.add_argument('--eval_batch_size', default =100, type=int, help= "")




parser.add_argument('--eval_restore', default =None, type=int, help= "")
parser.add_argument('--data_root', default ="/home/kim2712/Desktop/data", help= "Where data is sitting")






args = parser.parse_args()    

args.save_path = os.path.join(args.save_path,args.model)

pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)

def _log(message):
        print(message)
        f = open(os.path.join(args.save_path, 'evaluation_log.txt'), 'a+')
        f.write(message + '\n')
        f.close()



model = get_resnet_by_capa(args.capa)
ckpt = torch.load(args.model_path)

if args.model =="TRADE":
    model.load_state_dict(ckpt)
    model = nn.DataParallel(model).cuda()    
else:
    model = nn.DataParallel(model).cuda()    
    model.load_state_dict(ckpt['model'])

_log(F"{args.model} model loaded from {args.model_path}\n")      


model.eval()

for param in model.parameters():
    param.requires_grad = False




evaluator = Evaluator(args,model)
evaluator.eval()