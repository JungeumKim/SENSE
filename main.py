# Modified the templete in : https://github.com/yogeshbalaji/Instance_Adaptive_Adversarial_Training
from IPython.core.debugger import set_trace

import argparse
import json
from train.trainer_sense import TrainerSENSE
import sense.utils.utils_from_IAAT as utils
import os
import pathlib
from eval.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
    parser.add_argument('--cfg_path', default='train/train_sense.json', type=str,
                        help='path to config file')
    
    parser.add_argument('--data_root', default=None, type=str,
                        help='path to dataset')
    
    parser.add_argument('--alg', default=None, type=str,
                        help='[sense-at,nat,sense-trade]')
    
    parser.add_argument('--save_path', default=None, type=str,
                        help='path to save file')
        
    parser.add_argument('--mode', default=None,
                        help='mode to use| [train,eval]')
    parser.add_argument('--restore', default=None, help='path to restore')
    parser.add_argument('--eval_restore', default=None, help='name of a model to be evaluated in eval mode')    
    
    parser.add_argument('--train_sample_size', default=None,type=int, help='')
    parser.add_argument('--eval_sample_size', default=None,type=int, help='')
    parser.add_argument('--batch_size', default=None,type=int, help='')
    parser.add_argument('--test_sample_size', default=None,type=int, help='')
    parser.add_argument('--warmup', default=None,type=int, help='')
    parser.add_argument('--eval_batch_size', default=None,type=int, help='')
    parser.add_argument('--nat_weight', default=None,type=float, help='')
    parser.add_argument('--lossCut', default=None,type=float, help='')
    parser.add_argument('--nepochs', default=None,type=int, help='')
    parser.add_argument('--capa', default=None,type=int, help='')
    parser.add_argument('--fab_nb', default=None,type=int, help='')
    parser.add_argument('--demo', default = False, help ="", action ="store_true")
    args = parser.parse_args()
    return args


def main(args):

    # Read configs
    with open(args.cfg_path, "r") as fp:
        configs = json.load(fp)

    # Update the configs based on command line args
    arg_dict = vars(args)
    for key in arg_dict:
        if key in configs:
            if arg_dict[key] is not None:
                configs[key] = arg_dict[key]
                
                
    configs = utils.ConfigMapper(configs)

    configs.attack_eps = float(configs.attack_eps) / 255
    configs.attack_stepsize = float(configs.attack_stepsize) / 255

    if configs.alg =='nat': configs.lossCut=1

    if configs.mode == 'train':
        if configs.capa in range(-1,7): 
            configs.save_path = os.path.join(configs.save_path,F"capa{configs.capa}",["full","demo"][args.demo],configs.mode,configs.alg,F"c{configs.lossCut}",F"nat_weight{configs.nat_weight}")
        else:
            configs.save_path = os.path.join(configs.save_path,["full","demo"][args.demo],configs.mode,configs.alg,F"c{configs.lossCut}",F"nat_weight{configs.nat_weight}")
    
    
        pathlib.Path(configs.save_path).mkdir(parents=True, exist_ok=True)

        trainer = TrainerSENSE(configs)
        trainer.train()

    elif configs.mode == 'eval':
        evaluator = Evaluator(configs)
        evaluator.eval()

    else:

        raise ValueError(F'mode {configs.mode} not implemented')
    return configs.save_path


if __name__ == '__main__':
    args = parse_args()
    save_path = main(args)
    
    args.mode = "eval"
    args.save_path = save_path 
    main(args)


