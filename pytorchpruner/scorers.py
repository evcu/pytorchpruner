'''
Each scorer should implement score function, which given parameter tensor returns the
scores in the same shape. High scores mean high salincies.
'''

from .utils import hessian_fun,gradient_fun

def magnitudeScorer(params):
    return params.data.clone().abs()


def gradientScorer(loss,params):
    """
    Follows gradient_funs behaviour about list of params. An extension is expected
    """
    return gradient_fun(loss,params)

def hessianScorer(loss,params):
    """
    Follows hessian_funs behaviour about list of params. An extension is expected
    """
    ##TODO after fixing hessian_fun return behaviour update here
    #print(hessian.sum(1).sum(1).view(2,2))
    acc_hessian = hessian_fun(loss,params).sum(0)
    return acc_hessian
