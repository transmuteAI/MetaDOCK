import torch.nn as nn
import numpy as np
from .layers import PrunableConv2d

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.prunable_modules = []
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def n_remaining(self, m):
        return (m.get_binary_zetas()).sum()
    
    def is_all_pruned(self, m):
        return self.n_remaining(m) == 0

    def get_remaining(self):
        """return the fraction of active zeta (i.e > 0)""" 
        n_rem = 0
        n_total = 0
        for l_block in self.prunable_modules:
            n_rem += self.n_remaining(l_block)
            n_total += l_block.num_gates
        return (n_rem)/n_total

    def give_zetas(self):
        zetas = []
        for l_block in self.prunable_modules:
            zetas.append(l_block.zeta.cpu().detach().numpy().tolist())
        zetas = [z for k in zetas for z in k ]
        return zetas
    
    def calculate_threshold(self, budget):
        zetas = np.array(sorted(self.give_zetas()))
        # print(zetas.shape)
        threshold = zetas[int((1-budget)*len(zetas))]
        return threshold

    def give_pruned_zetas(self):
        zetas = []
        for l_block in self.prunable_modules:
            zetas.append(l_block.pruned_zeta.cpu().detach().numpy().tolist())
        zetas = [z for k in zetas for z in k ]
        return zetas

    def prune(self):
        """prunes the network to make zeta exactly 1 and 0"""
        for l_block in self.prunable_modules:
            l_block.prune()
        return None

    def unprune(self):
        for l_block in self.prunable_modules:
            l_block.unprune()

    def freeze_weights(self):
        self.requires_grad = False
        for l_block in self.prunable_modules:
            l_block.unprune()  

    def get_channels(self):
        total_channels = 0.
        active_channels = 0.
        for l_block in self.prunable_modules:
                active_channels+=l_block.pruned_zeta.sum().item()
                total_channels+=l_block.num_gates
        return active_channels, total_channels

    def get_params(self):
        pass

    def get_flops(self):
        pass