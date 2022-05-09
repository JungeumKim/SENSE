import os, sys, argparse
import torch
import time, datetime
from evaluator import Evaluator

sys.path.append("../")
from importlib import reload
import sense.utils.model_loading as ml

import pathlib
parser = argparse.ArgumentParser(description='Arguments.')
parser.add_argument('--save_path', default ="./result/wide_net_test_result", help= "")
parser.add_argument('--eval_sample_size', default =10000, type=int, help= "The first sample size of the 10000 test set is used")
parser.add_argument('--eval_group1', default = False, help ="", action ="store_true")
parser.add_argument('--eval_group2', default = False, help ="", action ="store_true")
parser.add_argument('--eval_group3', default = False, help ="", action ="store_true")
parser.add_argument('--eval_group4', default = False, help ="", action ="store_true")
parser.add_argument('--eval_group5', default = False, help ="", action ="store_true")
parser.add_argument('--demo', default = False, help ="", action ="store_true")
parser.add_argument('--fab_nb', default =1, type=int, help= "")
parser.add_argument('--eval_batch_size', default =100, type=int, help= "")

parser.add_argument('--capa', default =None, type=int, help= "")
parser.add_argument('--model_path', default =None, help= "A certian model to be evaluated")

parser.add_argument('--eval_restore', default =None, type=int, help= "")
parser.add_argument('--data_root', default ="/home/kim2712/Desktop/data", help= "Where data is sitting")


parser.add_argument('--model', default ="SENSE", help= "[SENSE,IAAT,MART,TRADE,MMA-12,-20,-32]")


args = parser.parse_args()    

args.save_path = os.path.join(args.save_path,args.model)

pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)

def _log(message):
        print(message)
        f = open(os.path.join(args.save_path, 'evaluation_log.txt'), 'a+')
        f.write(message + '\n')
        f.close()


model = ml.model_loader(args.model,"../trained_models",path=args.model_path)
model.eval()
message = F"{args.model} model loaded from {args.model_path}\n" if args.model_path is not None else F"{args.model} model loaded"
_log(message)
evaluator = Evaluator(args,model)
evaluator.eval()
