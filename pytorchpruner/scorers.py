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

def taylor1Scorer(params,loss,w_fun=lambda a: -a):
    """
    taylor1Scorer

    """
    if isinstance(params,nn.Parameter):
        dw = w_fun(params.data)
        grad = gradient_fun(loss,params,retain_graph=True).data
        result = torch.mul(dw,grad)
    elif isinstance(params,collections.Iterable):
        params = list(params)
        grads = gradient_fun(loss,params,retain_graph=True)
        result = list(map(lambda w,g:torch.mul(w_fun(w.data),g),zip(params,grads)))
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameter" % type(params))
    return result

def taylor1ScorerAbs(params,loss,w_fun=lambda a:-a):
    return taylor1Scorer(params,loss,w_fun=w_fun).abs()


def hessianScorer(params,loss,w_fun=lambda a:-a):
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
        vector = w_fun(params.data.clone())
        hv = hessian_vector_product(loss,params,vector,retain_graph=True)
        result = torch.mul(w_fun(params.data),hv)
    elif isinstance(params,collections.Iterable):
        # Case 2
        params = list(params)
        rev_f,n_elements = get_reverse_flatten_params_fun(params,get_count=True)
        vector = flatten_params((w_fun(p.data.clone()) for p in params))
        flat_hv = hessian_vector_product(loss,params,vector,retain_graph=True,flattened=True)
        hv = rev_f(flat_hv)
        result = list(map(lambda w,h:torch.mul(w_fun(w.data),h),zip(params,hv)))
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameter" % type(params))

    return result

def hessianScorerAbs(params,loss,w_fun=lambda a:-a):
    return hessianScorer(params,loss,w_fun=w_fun).abs()

def taylor2Scorer(params,loss,w_fun=lambda a: -a,scale=1):
    """
    taylor2Scorer

    """
    if not isinstance(scale, (int, float)):
        raise ValueError(f'scale={float} needs tobe a float or int')
    if isinstance(params,nn.Parameter):
        vector = w_fun(params.data.clone())
        hv = hessian_vector_product(loss,params,vector,retain_graph=True)
        grad = gradient_fun(loss,params,retain_graph=True).data
        result = torch.mul(w_fun(params.data),scale*hv+grad)
    elif isinstance(params,collections.Iterable):
        # Case 2
        params = list(params)
        rev_f,n_elements = get_reverse_flatten_params_fun(params,get_count=True)
        vector = flatten_params((w_fun(p.data.clone()) for p in params))
        flat_hv = hessian_vector_product(loss,params,vector,retain_graph=True,flattened=True)
        hv = rev_f(flat_hv)
        grads = gradient_fun(loss,params,retain_graph=True)
        result = list(map(lambda w,h,g:torch.mul(w_fun(w.data),scale*h+g),zip(params,hv,grads)))
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameter" % type(params))
    return result

def taylor2ScorerAbs(params,loss,w_fun=lambda a:-a,scale=1):
    return taylor2Scorer(params,loss,w_fun=w_fun,scale=scale).abs()

def lossChangeScorer(params,loss,loss_calc_f=None):
    if loss_calc_f is None:
        raise ValueError(f'loss_calc_f:{loss_calc_f} cannot be None')
    else:
        scores = params.data.clone()
        for idx in product(*map(range,scores.size())):
            old_val,params.data[idx] = params.data[idx],0
            scores[idx] = loss_calc_f()[0].data[0]-loss.data[0]
            params.data[idx] = old_val
    return scores

def lossChangeScorerAbs(*args,**kwargs):
    scores = lossChangeScorer(*args,**kwargs)
    return scores.abs()

def randomScorer(params,loss):
    old_state = torch.get_rng_state()
    res = torch.rand(params.size())
    torch.set_rng_state(old_state)
    return res
