import numpy as np
import torch
import utils
from utils import custom_grad, freeze_model, unfreeze_model
from models import get_conv_model

class Learner:
    def __init__(self, model, 
                 loss_function, 
                 inner_lr=1e-1, 
                 outer_lr=1e-3, 
                 GPU=False, 
                 outer_alg='adam', 
                 inner_alg='bilevel',
                 budget=0.5, 
                 w1=2,
                 w2=0,
                 num_total_iter=80000,
                 budget_scheduler_type=None,
                 budget_weight_scheduler_type=None):
        self.model = model
        self.use_gpu = GPU
        if GPU:
            self.model.cuda()
        self.device = next(self.model.parameters()).device
        self.inner_opt = torch.optim.SGD(self.model.parameters(), lr=inner_lr)
        self.inner_alg = inner_alg
        if outer_alg == 'adam':
            self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=outer_lr, eps=1e-3)
        else:
            self.outer_opt = torch.optim.SGD(self.model.parameters(), lr=outer_lr)
        self.loss_function = loss_function
        self.budget = budget
        self.bw = w1
        self.l1_weight = w2
        self.hist = {}
        self.curr_iter = 0
        self.num_total_iter = num_total_iter
        self.budget_scheduler_type = budget_scheduler_type
        self.budget_weight_scheduler_type = budget_weight_scheduler_type

    def get_params(self):
        return torch.cat([param.data.view(-1) for param in self.model.parameters()], 0).clone()

    def set_params(self, param_vals):
        offset = 0
        for param in self.model.parameters():
            param.data.copy_(param_vals[offset:offset + param.nelement()].view(param.size()))
            offset += param.nelement()
            
    def set_outer_lr(self, lr):
        for param_group in self.outer_opt.param_groups:
            param_group['lr'] = lr
            
    def set_inner_lr(self, lr):
        for param_group in self.inner_opt.param_groups:
            param_group['lr'] = lr
            
    def budget_loss(self):
        budget = self.budget_scheduler()
        Vc = torch.FloatTensor([budget])
        return ((self.model.get_remaining().to(self.device)-Vc.to(self.device))**2).to(self.device)

    def regularization_loss(self, w_0, lam=0.0):
        """
        Add a regularization loss onto the weights
        The proximal term regularizes around the point w_0
        Strength of regularization is lambda
        lambda can either be scalar (type float) or ndarray (numpy.ndarray)
        """
        regu_loss = 0.0
        offset = 0
        regu_lam = lam if type(lam) == float or np.float64 else utils.to_tensor(lam)
        if w_0.dtype == torch.float16:
            try:
                regu_lam = regu_lam.half()
            except:
                regu_lam = np.float16(regu_lam)
        for param in self.model.parameters():
            delta = param.view(-1) - w_0[offset:offset + param.nelement()].view(-1)
            if type(regu_lam) == float or np.float64:
                regu_loss += 0.5 * regu_lam * torch.sum(delta ** 2)
            else:
                # import ipdb; ipdb.set_trace()
                param_lam = regu_lam[offset:offset + param.nelement()].view(-1)
                param_delta = delta * param_lam
                regu_loss += 0.5 * torch.sum(param_delta ** 2)
            offset += param.nelement()
        return regu_loss

    def get_loss(self, x, y, return_numpy=False, use_budget=False):
        """
        Assume that x and y are torch tensors -- either in CPU or GPU (controlled externally)
        """
        yhat = self.model.forward(x)
        loss = self.loss_function(yhat, y)
        bw = self.budget_weight_scheduler()
        if use_budget:
            loss = loss + bw*self.budget_loss() + self.l1_weight*self.l1_on_zeta()
        if return_numpy:
            loss = utils.to_numpy(loss).ravel()[0]
        return loss
    
    def l1_on_zeta(self):
        loss = 0.0
        for name, param in self.model.named_parameters():
            if 'zeta' in name:
                loss = loss + torch.norm(param, 1)
        return loss
    
    def predict(self, x, return_numpy=False):
        yhat = self.model.forward(utils.to_device(x, self.use_gpu))
        if return_numpy:
            yhat = utils.to_numpy(yhat)
        return yhat

    def learn_on_data(self, x, y, num_steps=10,
                      add_regularization=False,
                      w_0=None, lam=0.0, inner_batch_size=None):
        
        train_loss = []
        if self.inner_alg == 'gradient':
            for i in range(num_steps):
                if inner_batch_size:
                    mask = np.random.randint(0, len(x), inner_batch_size)
                    xt, yt = x[mask], y[mask]
                else:
                    xt, yt = x, y
                self.inner_opt.zero_grad()
                tloss = self.get_loss(xt, yt)
                loss = tloss + self.regularization_loss(w_0, lam) if add_regularization else tloss
                loss.backward()
                self.inner_opt.step()
                train_loss.append(utils.to_numpy(tloss))
        elif self.inner_alg == 'bilevel':
            for i in range(num_steps):
                if inner_batch_size:
                    mask = np.random.randint(0, len(x), inner_batch_size)
                    xt, yt = x[mask], y[mask]
                else:
                    xt, yt = x, y
                self.inner_opt.zero_grad()
                self.model.prune()
                tloss = self.get_loss(xt, yt)
                loss = tloss + self.regularization_loss(w_0, lam) if add_regularization else tloss
                loss.backward()
                freeze_model(self.model)
                self.model.unprune()
                tloss2 = self.get_loss(xt, yt, use_budget=True)
                tloss2.backward()
                unfreeze_model(self.model)
                self.inner_opt.step()
                train_loss.append(utils.to_numpy(tloss))
        return train_loss

    def learn_task(self, xt, yt, num_steps=10, add_regularization=False, w_0=None, lam=0.0, inner_batch_size = None):
        return self.learn_on_data(xt, yt, num_steps, add_regularization, w_0, lam, inner_batch_size)

    def move_toward_target(self, target, lam=2.0):
        """
        Move slowly towards the target parameter value
        Default value for lam assumes learning rate determined by optimizer
        Useful for implementing Reptile
        """
        # we can implement this with the regularization loss, but regularize around the target point
        # and with specific choice of lam=2.0 to preserve the learning rate of inner_opt
        self.outer_opt.zero_grad()
        loss = self.regularization_loss(target, lam=lam)
        loss.backward()
        self.outer_opt.step()

    def outer_step_with_grad(self, grad, flat_grad=False):
        """
        Given the gradient, step with the outer optimizer using the gradient.
        Assumed that the gradient is a tuple/list of size compatible with model.parameters()
        If flat_grad, then the gradient is a flattened vector
        """
        check = 0
        for p in self.model.parameters():
            check = check + 1 if type(p.grad) == type(None) else check
        if check > 0:
            # initialize the grad fields properly
            dummy_loss = self.regularization_loss(self.get_params())
            dummy_loss.backward()  # this would initialize required variables
        if flat_grad:
            offset = 0
            grad = utils.to_device(grad, self.use_gpu)
            for p in self.model.parameters():
                this_grad = grad[offset:offset + p.nelement()].view(p.size())
                p.grad.copy_(this_grad)
                offset += p.nelement()
        else:
            for i, p in enumerate(self.model.parameters()):
                p.grad = grad[i]
        self.outer_opt.step()
        self.curr_iter+=1

    def matrix_evaluator(self, xt, yt, lam, regu_coef=1.0, lam_damping=10.0, x=None, y=None):
        """
        Constructor function that can be given to CG optimizer
        Works for both type(lam) == float and type(lam) == np.ndarray
        """
        if type(lam) == np.ndarray:
            lam = utils.to_device(lam, self.use_gpu)
        def evaluator(v):
            hvp = self.hessian_vector_product(xt, yt, v, x=x, y=y)
            Av = (1.0 + regu_coef) * v + hvp / (lam + lam_damping)
            return Av
        return evaluator

    def hessian_vector_product(self, xt, yt, vector, params=None, x=None, y=None):
        """
        Performs hessian vector product on the train set in task with the provided vector
        """
        if params is not None:
            self.set_params(params)
        tloss = self.get_loss(xt, yt)
        params = [i for i in self.model.parameters()]
        grad_ft = custom_grad(tloss, params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
        vec = utils.to_device(vector, self.use_gpu)
        h = torch.sum(flat_grad * vec)
        hvp = custom_grad(h, params)
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat
    
    def budget_scheduler(self):
        if self.budget_scheduler_type is None:
            return self.budget
        budget = max(self.budget, np.exp(-4*self.curr_iter/self.num_total_iter))
        return budget

    def budget_weight_scheduler(self):
        if self.budget_weight_scheduler_type is None:
            return self.bw
        bw = self.bw*self.curr_iter/self.num_total_iter
        return bw

def make_conv_network(out_dim, task='omniglot', model_name='4conv', method='full', mode='conv'):
    if '4conv' in model_name:
        expansion = model_name.split('_')[-1]
        if task == 'cifar_fs':
            model = get_conv_model(method, out_dim, 3, 32, expansion=float(expansion), mode=mode)
        elif task == 'miniimagenet':
            model = get_conv_model(method, out_dim, 3, 84, expansion=float(expansion), mode=mode)         
    return model