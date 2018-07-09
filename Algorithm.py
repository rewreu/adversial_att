import torch
import numpy as np

def fgsm(steps, model, criterion, variable, x, y, eps, alpha):
    """
    Fast sign gradient method, the steps arg is not used here. 
    """
    fc_out = model(variable)
    loss = criterion(fc_out, y)
    loss.backward()
    variable.data = int(np.sign(alpha))*eps*torch.sign(variable.grad.data)+variable.data
    variable.data = torch.clamp(variable.data, 0, 1)
    variable.grad.data.fill_(0)
    return variable

def i_fgsm(steps, model, criterion, variable, x, y, eps, alpha):
    for i in range(steps):
        fc_out = model(variable)
        loss = criterion(fc_out, y)
        loss.backward()
        variable.data = alpha*torch.sign(variable.grad.data)+variable.data
        variable.data = torch.where(variable.data>x+eps, x+eps, variable.data)
        variable.data = torch.where(variable.data<x-eps, x-eps, variable.data)
        variable.data = torch.clamp(variable.data, 0, 1)
        variable.grad.data.fill_(0)
    return variable

def deepfool(steps, model, criterion, variable, x, y, eps, alpha):
    num_classes, overshoot = 10, 0.02
    f_image = model.forward(x).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]
    w , r_tot= [np.zeros(x.shape)]*2
    
    x0 = torch.tensor(variable.data).requires_grad_()
    fs = model.forward(x0)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    for i in range(steps):
        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x0.grad.data.cpu().numpy().copy()
        for k in range(1, num_classes):
            x0.grad.data.fill_(0)
            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x0.grad.data.cpu().numpy().copy()
            w_k = cur_grad - grad_orig # set new w_k and new f_k
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
            if pert_k < pert:
                pert = pert_k
                w = w_k
        r_i =  (pert+1e-4) * w / np.linalg.norm(w) #1e-4 for numerical stability
        r_tot = np.float32(r_tot + r_i)
        variable = x + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        x0 = torch.tensor(variable.data).requires_grad_()
        fs = model.forward(x0)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        if k_i!= label: break
    return variable