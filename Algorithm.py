import torch
import numpy as np
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


import numpy as np
from torch.autograd import Variable
import copy
from torch.autograd.gradcheck import zero_gradients

def deepfool(steps, model, criterion, variable, x, y, eps, alpha):
    num_classes=10
    overshoot=0.02
    max_iter= steps
    net = model
    image = variable
    f_image = net.forward(image).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().detach().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = variable
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label
    while k_i == label and loop_i < max_iter:
        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()
        for k in range(1, num_classes):
            zero_gradients(x)
            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()
            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k
        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)
        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        loop_i += 1
    r_tot = (1+overshoot)*r_tot
    return pert_image