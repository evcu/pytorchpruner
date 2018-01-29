from .context import pytorchpruner
import pytest
import torch
from torch.autograd import Variable
from torch.nn import Parameter

TOLERANCE = 1e-5
from pytorchpruner.scorers import magnitudeScorer,gradientScorer,hessianScorer

def L2d(w):
    '''
    A custom loss function
        w00 w01
        w10 w11
    '''
    return (w[1,0]**2)*w[1,1]+4*(w[0,0]**3)*w[1,0]
def GScore(w):
    #Jacobian of the L(w)
    wd = w.data
    return torch.Tensor([
                         [12*(wd[0,0]**2)*wd[1,0],
                          0],
                         [2*wd[1,0]*wd[1,1]+4*(wd[0,0]**3),
                          wd[1,0]**2]
                        ])


def HScore(w):
    #Hessian of the L2(w)
    wd=w.data
    gw12 = 2*wd[1,0]
    gw13 = 12*(wd[0,0]**2)
    gw23 = 0
    gw11 = 2*wd[1,1]
    gw22 = 0
    gw33 = 24*wd[0,0]*wd[1,0]
    ## x3 0
    ## x1 x2
    return torch.Tensor([[gw33+0+gw13+gw23, #0,0,: #x3 with others
                       0+0+0+0], #0,1,:
                      [gw13+0+gw11+gw12, #1,0,: #x1 with others
                       gw23+0+gw12+gw22]]) #1,1,: #x2 with others

def test_gradientScorer():
    w = Parameter(torch.rand(2,2))

    loss_val = L2d(w)
    grad_score = gradientScorer(loss_val, w)
    #emprical score
    correct_score = GScore(w).abs()
    assert (grad_score-correct_score).abs().sum() < TOLERANCE

def test_hessianScorer():
    w = Parameter(torch.rand(2,2))
    loss_val = L2d(w)

    hessian_score = hessianScorer(loss_val,w)
    #emprical score
    correct_score = HScore(w).abs()
    assert (hessian_score-correct_score).abs().sum() < TOLERANCE
