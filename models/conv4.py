import torch
import torch.nn as nn
from torch.nn import functional as F
from .layers import ModuleInjection, PrunableConv2d
from .base_model import BaseModel

class Block(nn.Module):
    def __init__(self, kernel_size, stride, in_size, out_size, track_bn=False):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_size, affine=True, momentum=1., track_running_stats=track_bn)
        self.conv1, self.bn1 = ModuleInjection.make_prunable(self.conv1, self.bn1)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return x

    def get_flops(self, activation_h):
        sparse_mask = (self.conv1.zeta>0)*1.
        in_channels, output_channels = sparse_mask.shape[1], sparse_mask.shape[0]
        N_conv_ops = activation_h * activation_h / 2.
        Mac_per_filter_sparse = 9*torch.sum(sparse_mask)
        Mac_per_filter_dense = 9*in_channels*output_channels
        flops_bn_relu = 3*activation_h*activation_h*in_channels

        flops_sparse = 2*Mac_per_filter_sparse*N_conv_ops + flops_bn_relu
        flops_dense = 2*Mac_per_filter_dense*N_conv_ops + flops_bn_relu
        return flops_sparse, flops_dense

    def get_params(self):
        sparse_mask = (self.conv1.zeta>0)*1.
        in_channels, output_channels = sparse_mask.shape[1], sparse_mask.shape[0]
        params_conv_sparse = 9*torch.sum(sparse_mask)
        params_conv_dense = 9*in_channels*output_channels
        params_bn = 2*output_channels
        params_sparse = params_conv_sparse + params_bn
        params_dense = params_conv_dense + params_bn
        return params_sparse, params_dense
    
class Conv4(BaseModel):
    """
    A 4 layer model for CIFAR/Omniglot.
    """
    def __init__(self, in_size, in_channels, num_classes, hidden_size=64, feature_size=64, non_transductive=False, expansion=1):
        super(Conv4, self).__init__()

        self.input_size = in_size
        self.num_classes = num_classes
        self.num_filters = hidden_size
        self.out_features = feature_size
        self.in_channels = in_channels
        self.track_bn = non_transductive
        self.expansion = expansion

        self.layer = self._make_layer(block = Block)
        if self.input_size==32:
            self.classifier = nn.Sequential(
                nn.Linear(int(self.out_features*2*2*self.expansion), self.num_classes),
            )
        elif self.input_size==84:
            self.classifier = nn.Sequential(
                nn.Linear(int(self.out_features*6*6*self.expansion), self.num_classes),
            )
        else:
            raise ValueError('input size not supported')

    def _make_layer(self, block):
        layers = []
        layers.append(block(3, 2, self.in_channels, int(self.num_filters*self.expansion), self.track_bn))
        layers.append(block(3, 2, int(self.num_filters*self.expansion), int(self.num_filters*self.expansion), self.track_bn))
        layers.append(block(3, 2, int(self.num_filters*self.expansion), int(self.num_filters*self.expansion), self.track_bn))
        layers.append(block(3, 2, int(self.num_filters*self.expansion), int(self.num_filters*self.expansion), self.track_bn))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        out = out.view(len(out), -1)
        out = self.classifier(out)
        return out

    def predict(self, prob):
        _, argmax = prob.max(1)
        return argmax

    def get_flops(self):
        activation_h = self.input_size
        flops_sparse = 0
        flops_dense = 0
        for module in self.layer:
            flops_sparse_, flops_dense_ = module.get_flops(activation_h)
            flops_sparse += flops_sparse_
            flops_dense += flops_dense_
            activation_h = (activation_h+1)//2
        flops_linear = 2*self.classifier[0].in_features*self.classifier[0].out_features + self.classifier[0].out_features
        return flops_sparse+flops_linear, flops_dense+flops_linear
    
    def get_params(self):
        params_sparse = 0
        params_dense = 0
        for module in self.layer:
            params_sparse_, params_dense_ = module.get_params()
            params_sparse += params_sparse_
            params_dense += params_dense_
        params_linear = self.classifier[0].in_features*self.classifier[0].out_features + self.classifier[0].out_features
        return params_sparse+params_linear, params_dense+params_linear
    
def get_conv_model(method, num_classes, inchannels, insize, expansion, mode = 'conv'):
    ModuleInjection.pruning_method = method
    ModuleInjection.mode = mode
    ModuleInjection.prunable_modules = []
    net = Conv4(insize, inchannels, num_classes, expansion=expansion)
    net.prunable_modules = ModuleInjection.prunable_modules
    return net