'''
Each scorer should implement score function, which given parameter tensor returns the
scores in the same shape. High scores mean high salincies.
'''

from .utils import hessian_fun,gradient_fun,get_reverse_flatten_params_fun,hessian_vector_product
from torch import nn
import collections
import torch
def magnitudeScorer(params):
    return params.data.clone().abs()


def gradientScorer(loss,params):
    """
    Follows gradient_funs behaviour about list of params.
    """
    return gradient_fun(loss,params,retain_graph=True).data.abs()

def hessianScorer(loss,params):
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
