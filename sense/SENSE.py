#from IPython.core.debugger import set_trace

import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from advertorch.utils import normalize_by_pnorm

from .utils.utils_sense import delta_initialization_random, clipper



class SENSE:
    '''
    attack_norm = [float("inf"),2]
    
    usage: 
    Origianl SENSE-AT: perturb(x,y)
    TRADE type, SENSE-TRADE: perturb(x,y,Py) where Py=model(x): Py must be attached in the gradient map
    
    '''
    def __init__(self, model, epsilon,a,K, lossCut, attack_norm=float("inf"),attack_rand_start=True):
        self.epsilon = epsilon
        self.a = a
        self.K = K
        self.attack_rand_start = attack_rand_start
        
        assert attack_norm in [float("inf"), 2], "attack norm sould be either Linfty or L2"
        self.norm = attack_norm
        
        self.Cut = -np.log(lossCut)
        self.cross_vec = nn.CrossEntropyLoss(reduction='none')
        self.kl_mat = nn.KLDivLoss(reduction='none')

        self.model = model       
    
    def trade_kl(self, adv_output,Py):
        loss_kl = self.kl_mat(F.log_softmax(adv_output, dim=1), F.softmax(Py, dim=1))
        return loss_kl.sum(1)
        
    def perturb(self, X, y,Py=None,adaptive=True): 
        '''
        Usage:
        Py = model(X) : it is a combination of SENSE-TRADE, maximize loss that is KL(pred(x), pred(x-adv)).
        
        adaptive = True : normal sensible adv. example with three stages: nat, adpative, full PGD
        adaptive = False: two stages: nat(for initially ignored points), full PGD (for initially not ignored points)
        Initial non-ignoring is decided by correct and with natural loss<self.Cut
        
        '''

        if self.Cut == 0 or self.K==0 or self.epsilon == 0 : # Natural learning
            return X  
        
        model_were_training = self.model.training
        if model_were_training: self.model.eval()
            
        for param in self.model.parameters():
            param.requires_grad = False
        
        n=y.shape[0]

        if self.attack_rand_start:
            delta=delta_initialization_random(X, p=self.norm, epsilon=self.epsilon)
            Xadv = clipper(X+delta, X,p=self.norm,epsilon=self.epsilon) 
            delta = Xadv-X
        else:
            delta = torch.zeros_like(X)
        delta.requires_grad_()
        
        adv_output = self.model(X+delta) 
        if Py is not None: Py = Py.detach().clone()
        
        loss_vec = self.cross_vec(adv_output, y) 

        with torch.no_grad():
            pred = adv_output.max(1, keepdim=True)[1]

            # No adversarial perturbations on x that is origianlly incorrectly classified or natural_loss>thresh
            Ignore = ~pred.eq(y.view_as(pred)).view(-1)  | (loss_vec > self.Cut) 

        for i in range(self.K):
            
            if Ignore.sum()==n: break

            #By using adv_output in the previous iteration, we can save one feed-forward. 
            #Using adv_output is idential to use f(Xadv) for newly updated Xadv becuase loss_vec*(~Ignore) below.
            
            loss = (loss_vec*(~Ignore).float()).mean()
            loss.backward()

            # note: delta.grad are zero for i s.t. ignore==True. 
            if self.norm == float("inf") :
                step = self.a*delta.grad.sign() 
            elif self.norm==2: 
                step = self.a*normalize_by_pnorm(delta.grad, p=2, small_constant=1e-6)
                
            with torch.no_grad():
                #project on the epsilon-ball and on the data range (0,1)
                intermed_delta=delta+step
                Xadv = clipper(X+intermed_delta, X,p=self.norm,epsilon=self.epsilon) 
                delta = Xadv-X
                delta.requires_grad_()
 
            adv_output  = self.model(X+delta) 
            loss_vec = self.cross_vec(adv_output, y) if Py is None else self.trade_kl(adv_output, Py)         
            
            # Note for above : Deciding who will be sensibly reversed is based on clipped_Xadv.
            # The grad of adv_output will be used in the next iteration to get loss. 
            # It can be used because although clipped_Xadv is not equal to Xadv for indices Ignore==True, 
            # they don't affact the update of Xadv.
           
            # Important: Below calc. does not have impact on the previous gradient map of loss_vec 
            # becasue delta.data is manupulated directly. 
            # Above backward still can get the effect of none-ignored delta values.
            
            with torch.no_grad():
                # Update ignoring indices based on the loss of the prediction on the clipped_Xadv
                if adaptive:
                    Ignore = Ignore | (loss_vec > self.Cut)                
                # Sensible reversion. for Ignore == 1, step back to previous delta.
                delta.data = delta.data.mul((~Ignore.view(n,1,1,1)).float())+intermed_delta.add(-step).mul(Ignore.view(n,1,1,1).float())
           
                                                 
        for param in self.model.parameters():
            param.requires_grad = True
            
        if model_were_training: self.model.train()
        delta.requires_grad_(False)
        
        return X+delta
    
    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)

