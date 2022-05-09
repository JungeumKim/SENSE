import torch
import sys
import os
from advertorch_examples.models import get_cifar10_wrn28_widen_factor
import torch.nn as nn

from ..others import models # for IAAT
from ..others import wider_resnet_coded_by_yaodongyu as yaodongyu #for TRADE, sense
from ..others import wideresnet as martlib #for MART

    


def model_loader(model,model_home, path=None, device="cuda"):
    '''
    argument: 
    model: ["SENSE","IAAT","TRADE","MMA-12","MART"]
    '''
    sense = model_home+"/wider_net/c0.5/checkpoint.pth"
    iaat = model_home+"/BaseIineMethods/IAAT/checkpoint.pth"
    trade = model_home+"/BaseIineMethods/TRADES/checkpoint.pth"
    mart = model_home+"/BaseIineMethods/MART/checkpoint.pth"
    mma = model_home+"/BaseIineMethods/MMA12/checkpoint.pth"

    if "SENSE" in model:
        if path is not None: 
            sense = path
     
        model = yaodongyu.WideResNet() 
        model = nn.DataParallel(model).to(device)
        ckpt =  torch.load(sense)
        model.load_state_dict(ckpt['model'])
        
    elif "MMA" in model:
        ckpt = torch.load(mma)
        model = get_cifar10_wrn28_widen_factor(4)
        model.load_state_dict(ckpt["model"])
        model = torch.nn.DataParallel(model).to(device)
        
    elif model == "IAAT":        
        model = models.WRN(depth=32, width=10, num_classes=10)
        model = torch.nn.DataParallel(model).to(device)
        model_data = torch.load(iaat)
        model.load_state_dict(model_data['model'])

    elif model == "MART":
        model = martlib.WideResNet()
        model = torch.nn.DataParallel(model).to(device)
        ckpt = torch.load(mart)
        model.load_state_dict(ckpt)

    elif model == "TRADE":
        
        model = yaodongyu.WideResNet() 
        model.load_state_dict(torch.load(trade))
        model = nn.DataParallel(model).to(device)

    else:
        raise(F"No {args.model} is loaded!")
    
    return model
