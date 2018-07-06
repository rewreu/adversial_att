import torch
import numpy as np
def i_fgsm(steps, model, criterion, Variable, x, y, eps, alpha):
    for i in range(steps):
        fc_out = model(Variable)
        loss = criterion(fc_out, y)
        loss.backward()
        Variable.data = alpha*torch.sign(Variable.grad.data)+Variable.data
        Variable.data = torch.where(Variable.data>x+eps, x+eps, Variable.data)
        Variable.data = torch.where(Variable.data<x-eps, x-eps, Variable.data)
        Variable.data = torch.clamp(Variable.data, 0, 1)
        Variable.grad.data.fill_(0)
    return Variable

def fgsm(steps, model, criterion, Variable, x, y, eps, alpha):
    """
    Fast sign gradient method, the steps arg is not used here. 
    """
    fc_out = model(Variable)
    loss = criterion(fc_out, y)
    loss.backward()
    Variable.data = int(np.sign(alpha))*eps*torch.sign(Variable.grad.data)+Variable.data
    Variable.data = torch.clamp(Variable.data, 0, 1)
    Variable.grad.data.fill_(0)
    return Variable