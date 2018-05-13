'''
Each scorer should implement score function, which given parameter tensor returns the
scores in the same shape. High scores mean high salincies.
'''

from .modules import meanOutputReplacer
from torch import nn
import torch
#loss is not needer, but adding it here to have a generic set of parameters
def normScorer(layer,p=1,**kwargs):
    if isinstance(layer,meanOutputReplacer):
        layer = layer.module
    if isinstance(layer,(nn.Conv2d,nn.Linear)):
        normed_param = layer.weight.data.norm(p=p,dim=1)
        while normed_param.dim()>1:
            normed_param = normed_param.norm(p=p,dim=1)
        return normed_param
    else:
        raise ValueError("Invalid type, received: %s. should be a nn.Conv2d or nn.Linear" % type(layer))

def normScorerL1(layer,**kwargs):
    return normScorer(layer,p=1,**kwargs)

def normScorerL2(layer,**kwargs):
    return normScorer(layer,p=2,**kwargs)

def randomScorer(layer,p=1,**kwargs):
    old_state = torch.get_rng_state()
    if isinstance(layer,meanOutputReplacer):
        layer = layer.module
    if isinstance(layer,(nn.Conv2d,nn.Linear)):
        res = torch.rand(layer.weight.data.size(0))
        torch.set_rng_state(old_state)
        return res
    else:
        raise ValueError("Invalid type, received: %s. should be a nn.Conv2d or nn.Linear" % type(layer))

def mrsScorer(layer):
    if isinstance(layer,(meanOutputReplacer)):
        # print(f'is_mr:{layer.is_mean_replace},enabled:{layer.enabled}')
        return layer.mrss
    else:
        raise ValueError("Invalid type, received: %s. should be wrapped with meanOutputReplacer" % type(layer))

def mrpScorer(layer,loss_diff_f=None):
    if loss_diff_f is None:
        raise ValueError(f'loss_diff_f:{loss_diff_f} cannot be None')
    if isinstance(layer,(meanOutputReplacer)):
        n_units = layer.weight.data.size(0)
        dat_diffs = torch.zeros(n_units)
        old_state = layer.is_mean_replace
        layer.is_mean_replace = True
        for ui in range(n_units):
            layer.unit_id = ui
            dat_diffs[ui]= loss_diff_f()
        layer.is_mean_replace = old_state
        return dat_diffs
    else:
        raise ValueError("Invalid type, received: %s. should be wrapped with meanOutputReplacer" % type(layer))
