import torch

def batch_individual_norm(X, p): #learned from advertorch
    return X.norm(p=p,dim=3,keepdim=True).norm(p=p,dim=2,keepdim=True).norm(p=p,dim=1,keepdim=True)

def delta_initialization_random(X, p, epsilon): #learned from advertorch
    if type(epsilon)==type(None): 
        return 0.001 * torch.randn_like(X)
    delta = torch.zeros_like(X)
    if p in [1,2]:
        delta.data.uniform_(0, 1)
        delta.data=delta.data-X
        norm=batch_individual_norm(delta.data, p)
        delta.data=delta.data*torch.min(epsilon / norm, torch.ones_like(norm))
    else: 
        delta.data.uniform_(-epsilon,epsilon)
    return delta

def clipper(Xadv, X,p,epsilon):  
    device=X.device
    # 1. No restriction case. Only consider the data_range (0,1)        
    if epsilon==None: 
        clipped_Xadv=Xadv.clamp(0,1)
        clipped_Xadv.requires_grad = True
        return clipped_Xadv

    # 2. Restriction case. Only consider the data_range (0,1)        
    # projection of delta on the epsilon-ball 
    delta = Xadv - X
    if p==float("inf"):
        delta.clamp_(-epsilon, epsilon)
    elif p==2:  
        norm=batch_individual_norm(delta.data, p)
        delta.data=delta.data*torch.min(epsilon / norm, torch.ones_like(norm))
    else: 
        raise NotImplemented()
    Xadv = X + delta.to(device)   
    Xadv.clamp_(0,1)  
    return Xadv 