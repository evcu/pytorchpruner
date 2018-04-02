'''
Each scorer should implement score function, which given parameter tensor returns the
scores in the same shape. High scores mean high salincies.
'''

from .utils import hessian_fun,gradient_fun,get_reverse_flatten_params_fun,hessian_vector_product
from torch import nn
import collections
import torch
from itertools import product
#loss is not needer, but adding it here to have a generic set of parameters
def magnitudeScorer(params,*args,**kwargs):
    if isinstance(params,nn.Parameter):
        result = params.data.clone().abs()
    elif isinstance(params,collections.Iterable):
        # Case 2
        result = list(map(lambda x:x.data.clone().abs(),params))
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameter" % type(params))
    return result

def gradientScorer(params,loss):
    """
    Follows gradient_funs behaviour about list of params.
    """
    if isinstance(params,nn.Parameter):
        dat = params.data.clone()
        grad = gradient_fun(loss,params,retain_graph=True).data.abs()
        result = torch.mul(dat,grad)
    elif isinstance(params,collections.Iterable):
        # Case 2
        params = list(params)
        result = gradient_fun(loss,params,retain_graph=True)
        result = list(map(lambda x:x.data.abs(),result))
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameter" % type(params))
    return result

def hessianScorer(params,loss):
    """
    hessian Scorer which basically returns the sum of the row of the hessian
    using efficient hessian-vector product.

    params:
        single nn.Parameter     -> trivial
        iterator of Parameter's -> In this case we flattened the parameters
    returns:
        single nn.Parameter     -> a single score tensor same size as the input Parameter
        iterator of Parameter's -> an iterator of scores each is same size as the input Parameters

    Example:
        check `test_hessianScorer`
    """
    if isinstance(params,nn.Parameter):
        vector = torch.ones(params.size())
        hessian_score = hessian_vector_product(loss,params,vector,retain_graph=True).abs()
    elif isinstance(params,collections.Iterable):
        # Case 2
        params = list(params)
        rev_f,n_elements = get_reverse_flatten_params_fun(params,get_count=True)
        vector = torch.ones(n_elements)
        flat_hessian_score = hessian_vector_product(loss,params,vector,retain_graph=True,flattened=True).abs()
        hessian_score = rev_f(flat_hessian_score)
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameter" % type(params))

    return hessian_score


def taylor1Scorer(params,loss,pruning_mode=False,take_abs=False):
    """
    Follows gradient_funs behaviour about list of params.
    """
    grads = gradientScorer(params,loss)
    weigts = magnitudeScorer(params)
    def final_fun(g,w):
        if pruning_mode:
            result = torch.mul(g,-w)
        else:
            result = torch.mul(g,-w.sign())
        if take_abs:
            result = result.abs()

    if isinstance(params,nn.Parameter):
        result = final_fun(grads,weights)
    elif isinstance(params,collections.Iterable):
        result = list(map(final_fun,zip(grads,weights)))
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameter" % type(params))
    return result

def taylor1ScorerAbs(params,loss):
    return taylor1Scorer(params,loss,take_abs=True)

def taylor1ScorerPruningMode(params,loss):
    return taylor1Scorer(params,loss,pruning_mode=True)

def taylor1ScorerPruningModeAbs(params,loss):
    return taylor1Scorer(params,loss,pruning_mode=True,take_abs=True)

#TODO pruning mode(possibly other mode, too) is not exacctly true hessianScorer needs to get optional vector parameter. Implement and test it.
def taylor2Scorer(params,loss,scale=0.01,pruning_mode=False,take_abs=False):
    """
    Follows gradient_funs behaviour about list of params.

    """
    if not isinstance(scale, (int, float)):
        raise ValueError(f'scale={float} needs tobe a float or int')
    grads = gradientScorer(params,loss)
    hessians = hessianScorer(params,loss)
    weigts = magnitudeScorer(params)
    def final_fun(h,g,w):
        if pruning_mode:
            result = torch.mul(w,-scale*g+(scale**2)*h)
        else:
            result = torch.mul(w.sign(),-scale*g+(scale**2)*h)
        if take_abs:
            result = result.abs()

    if isinstance(params,nn.Parameter):
        result = final_fun(grads,weights)
    elif isinstance(params,collections.Iterable):
        result = list(map(final_fun,zip(grads,weights)))
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameter" % type(params))
    return result

def taylor2ScorerAbs(params,loss):
    return taylor2Scorer(params,loss,take_abs=True)

def taylor2ScorerPruningMode(params,loss):
    return taylor2Scorer(params,loss,pruning_mode=True)

def taylor2ScorerPruningModeAbs(params,loss):
    return taylor2Scorer(params,loss,pruning_mode=True,take_abs=True)

# TODO checkk that this is equal to taylor2Scorer and remove
def gradientDescentScorer(params,loss,scale=1):
    """
    check that
    """
    if not isinstance(scale, (int, float)):
        raise ValueError('scale={} needs to be a float or int'.format(float))
    if isinstance(params,nn.Parameter):
        grad_tensor = gradient_fun(loss,params,retain_graph=True).data.clone()
        hv = scale*hessian_vector_product(loss,params,grad_tensor,retain_graph=True)
        second_order_appx = torch.abs(-scale*torch.mul(grad_tensor,grad_tensor)
                                      +(scale**2)*torch.mul(grad_tensor,hv))
    elif isinstance(params,collections.Iterable):
        params = list(params)
        rev_f,n_elements = get_reverse_flatten_params_fun(params,get_count=True)
        flat_grad_tensor = gradient_fun(loss,params,retain_graph=True,flattened=True).data.clone()
        flat_hv = hessian_vector_product(loss,params,flat_grad_tensor,retain_graph=True,flattened=True).abs()
        flattened_second_order_appx = torch.abs(-scale*torch.mul(flat_grad_tensor,flat_grad_tensor)
                                       +(scale**2)*torch.mul(flat_grad_tensor,flat_hv))
        second_order_appx = rev_f(flattened_second_order_appx)
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameter" % type(params))
    return second_order_appx

def lossChangeScorer(params,loss,loss_calc_f=None):
    if loss_calc_f is None:
        raise ValueError(f'loss_calc_f:{loss_calc_f} cannot be None')
    else:
        scores = params.data.clone()
        for idx in product(*map(range,scores.size())):
            # import pdb;pdb.set_trace()
            old_val,params.data[idx] = params.data[idx],0

            scores[idx] = loss_calc_f()[0].data[0]-loss.data[0]
            params.data[idx] = old_val
    return scores
