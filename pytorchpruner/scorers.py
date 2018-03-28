'''
Each scorer should implement score function, which given parameter tensor returns the
scores in the same shape. High scores mean high salincies.
'''

from .utils import hessian_fun,gradient_fun,get_reverse_flatten_params_fun,hessian_vector_product
from torch import nn
import collections
import torch

#loss is not needer, but adding it here to have a generic set of parameters
def magnitudeScorer(loss,params,scale=1):
    if isinstance(params,nn.Parameter):
        result = params.data.clone().abs()
    elif isinstance(params,collections.Iterable):
        # Case 2
        result = list(map(lambda x:x.data.clone().abs(),params))
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameter" % type(params))
    return scale*result

def gradientScorer(loss,params):
    """
    Follows gradient_funs behaviour about list of params.
    """
    if isinstance(params,nn.Parameter):
        result = gradient_fun(loss,params,retain_graph=True).data.abs()
    elif isinstance(params,collections.Iterable):
        # Case 2
        params = list(params)
        result = gradient_fun(loss,params,retain_graph=True)
        result = list(map(lambda x:x.data.abs(),result))
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameter" % type(params))
    return result

def hessianScorer(loss,params,scale=1):
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
    if not isinstance(scale, (int, float)):
        raise ValueError('scale={} needs tobe a float or int'.format(float))
    if isinstance(params,nn.Parameter):
        vector = torch.ones(params.size())
        hessian_score = scale*hessian_vector_product(loss,params,vector,retain_graph=True).abs()
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

def gradientDescentScorer(loss,params,scale=1):
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
