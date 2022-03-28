import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle5 as pickle
import random
import torch
from torch.types import _TensorOrTensors
from typing import Optional, Tuple

def custom_grad(
    outputs: _TensorOrTensors,
    inputs: _TensorOrTensors,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False
) -> Tuple[torch.Tensor, ...]:
    parmas_index = [i for i,j in enumerate(inputs) if j.requires_grad]
    params_no_grad_shapes = [i.shape for i in inputs if not i.requires_grad]
    params_requiring_grad = [i for i in inputs if i.requires_grad]
    grads = torch.autograd.grad(outputs, params_requiring_grad, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph, only_inputs = only_inputs)
    grads_ = []
    j = 0
    k = 0
    for i in range(len(params_no_grad_shapes)+len(params_requiring_grad)):
        if i in parmas_index:
            grads_.append(grads[j])
            j+=1
        else:
            grads_.append(torch.zeros(params_no_grad_shapes[k], device=params_requiring_grad[0].device))
            k+=1
    return grads_
    

def to_cuda(x):
    try:
        return x.cuda()
    except:
        return torch.from_numpy(x).float().cuda()


def to_tensor(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x).float()
    elif type(x) == torch.Tensor:
        return x
    else:
        print("Type error. Input should be either numpy array or torch tensor")
    

def to_device(x, GPU=False):
    if GPU:
        return to_cuda(x)
    else:
        return to_tensor(x)
    
    
def to_numpy(x):
    if type(x) == np.ndarray:
        return x
    else:
        try:
            return x.data.numpy()
        except:
            return x.cpu().data.numpy()
        

def cg_solve(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10, x_init=None):
    """
    Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
    Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
    Algorithm template from wikipedia
    Verbose mode works only with numpy
    """
       
    if type(b) == torch.Tensor:
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax(x)
        p = r.clone()
    elif type(b) == np.ndarray:
        x = np.zeros_like(b) if x_init is None else x_init
        r = b - f_Ax(x)
        p = r.copy()
    else:
        print("Type error in cg")

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
            norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
            print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = f_Ax(p)
        alpha = rdotr/(p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr/rdotr
        p = r + beta * p

        if newrdotr < residual_tol:
            # print("Early CG termination because the residual was small")
            break

    if callback is not None:
        callback(x)
    if verbose: 
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x


def smooth_vector(vec, window_size=25):
    svec = vec.copy()
    if vec.shape[0] < window_size:
        for i in range(vec.shape[0]):
            svec[i,:] = np.mean(vec[:i, :], axis=0)
    else:   
        for i in range(window_size, vec.shape[0]):
            svec[i,:] = np.mean(vec[i-window_size:i, :], axis=0)
    return svec

    
def measure_accuracy(x, y, model):
    y_hat = model.predict(x, return_numpy = True)
    batch_size = y.shape[0]
    predict_label = np.argmax(y_hat, axis=1)
    try:
        correct = np.sum(predict_label == y.cpu().data.numpy())
    except:
        correct = np.sum(predict_label == y.data.numpy())
    return correct * 100.0 / batch_size

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
        
def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
        
def save_obj(obj, name, path='.'):
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, name+'.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, path='.'):
    file_path = os.path.join(path, name+'.pkl')
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def seed_all(seed=43):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def custom_loss_grad(x, y, fast_learner):
    fast_learner.model.prune(
        
    )
    loss1 = fast_learner.get_loss(x, y)
    params = [p for p in fast_learner.model.parameters()]
    valid_grad1 = custom_grad(loss1, params)
    freeze_model(fast_learner.model)
    fast_learner.model.unprune()
    loss2 = fast_learner.get_loss(x, y, use_budget=True)
    params = [p for p in fast_learner.model.parameters()]
    valid_grad2 = custom_grad(loss2, params)
    grads = [i+j for i,j in zip(valid_grad1, valid_grad2)]
    unfreeze_model(fast_learner.model)
    return loss1, grads

def combinations_custom(path_to_pickle):
    comb =  pickle.load(open(path_to_pickle, 'rb'))
    i=-1
    while i+1<len(comb):
        i+=1
        yield tuple(comb[i])

class StaticSampler():
    def __init__(self, path_to_pickle, size=600):
        self.combinations = combinations_custom(path_to_pickle)
        self.size = size

    def __iter__(self):
        return self.combinations
    
    def __len__(self):
        return self.size
