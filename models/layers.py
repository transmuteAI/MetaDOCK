import torch
import torch.nn as nn

def generalized_logistic(x, beta):
    return torch.sigmoid(beta*x)

def continous_heavy_side(x, gamma):
    return 1-torch.exp(-gamma*x)+x*torch.exp(-gamma)

class PrunableConv2d(nn.Conv2d):
    def __init__(self,
         conv2d_module
    ):
        super(PrunableConv2d, self).__init__(conv2d_module.in_channels, conv2d_module.out_channels, conv2d_module.kernel_size[0],
                                             conv2d_module.stride[0], conv2d_module.padding[0], conv2d_module.dilation[0],
                                             conv2d_module.groups, conv2d_module.bias != None,
                                             conv2d_module.padding_mode)
        self.is_pruned = False
        self.num_gates = self.in_channels*self.out_channels
        self.zeta = nn.Parameter(torch.rand(self.out_channels,self.in_channels) * 0.01)
        self.pruned_zeta = torch.ones_like(self.zeta)
        conv2d_module.register_forward_hook(fo_hook)
        beta=1.
        gamma=2.
        for n, x in zip(('beta', 'gamma'), (torch.tensor([x], requires_grad=False) for x in (beta, gamma))):
            self.register_buffer(n, x)   
    

    def forward(self, input):
        z = self.pruned_zeta if self.is_pruned else self.get_binary_zetas()
        w = self.weight*z[:,:, None, None]
        return self._conv_forward(input, w)

    def get_zeta_i(self):
        return generalized_logistic(self.zeta, self.beta)

    def get_zeta_t(self):
        zeta_i = self.get_zeta_i()
        return zeta_i

    def get_binary_zetas(self):
        return Binarize().apply(self.get_zeta_t())

    def set_beta_gamma(self, beta, gamma):
        self.beta.data.copy_(torch.Tensor([beta]))
        self.gamma.data.copy_(torch.Tensor([gamma]))

    def prune(self):
        self.is_pruned = True
        self.pruned_zeta = (self.zeta.detach()>0).float()
        self.zeta.requires_grad = False

    def unprune(self):
        self.is_pruned = False
        self.zeta.requires_grad = True

    @staticmethod
    def from_conv2d(bn_module, conv_module):
        new_conv = PrunableConv2d(conv_module)
        return bn_module, new_conv


def fo_hook(module, in_tensor, out_tensor):
    module.num_input_active_channels = (in_tensor[0].sum((0,2,3))>0).sum().item()
    module.output_area = out_tensor.size(2) * out_tensor.size(3)

class ModuleInjection:
    pruning_method = 'full'
    mode = 'conv'
    prunable_modules = []

    @staticmethod
    def make_prunable(conv_module, bn_module):
        """Make a (conv, bn) sequence prunable.
        :param conv_module: A Conv2d module
        :param bn_module: The BatchNorm2d module following the Conv2d above
        :param prune_before_bn: Whether the pruning gates will be applied before or after the Batch Norm
        :return: a pair (conv, bn) that can be trained to
        """
        if ModuleInjection.pruning_method == 'full':
            return conv_module, bn_module
        else:
            bn_module, conv_module = PrunableConv2d.from_conv2d(bn_module, conv_module)
            ModuleInjection.prunable_modules.append(conv_module)
        return conv_module, bn_module

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = (input>0.5).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None