import torch

def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False
def defreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
def is_freezed_all(model):
    return all([ not param.requires_grad for param in model.parameters()])

def is_defreezed_all(model):
    return all([ param.requires_grad for param in model.parameters()])

        
def save_model(model, path,verbose = False):
    torch.save({'state_dict': model.state_dict()}, path)
    if verbose:
        print("model is saved at "+ path)

    
def predict(logit):
        return logit.max(dim=1, keepdim=False)[1]
    