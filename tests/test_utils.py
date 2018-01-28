from .context import pytorchpruner
import pytest
import torch
from torch.autograd import Variable
from torch.nn import Parameter

TOLERANCE = 1e-5
from pytorchpruner.utils import hessian_fun,gradient_fun,flatten_params

def test_gradient_fun_1d():
    w = Parameter(torch.rand(3))
    def L(w):
        #A custom loss function
        return (w[0]**2)*w[1]+4*(w[2]**3)*w[0]
    def G(w):
        #Jacobian of the L(w)
        wd = w.data
        return torch.Tensor([2*wd[0]*wd[1]+4*(wd[2]**3),
                             wd[0]**2,
                            12*(wd[2]**2)*wd[0]])

    loss_val = L(w)
    autograd_grad = gradient_fun(loss_val, w).data
    #emprical gradient
    correct_grad = G(w)
    assert (autograd_grad-correct_grad).abs().sum() < TOLERANCE

def test_gradient_fun_2d():
    # We have 4 random parameters
    w = Parameter(torch.rand(2,2))
    def L2d(w):
        '''
        A custom loss function
            w00 w01
            w10 w11
        '''
        return (w[1,0]**2)*w[1,1]+4*(w[0,0]**3)*w[1,0]
    def G2d(w):
        #Jacobian of the L(w)
        wd = w.data
        return torch.Tensor([
                             [12*(wd[0,0]**2)*wd[1,0],
                              0],
                             [2*wd[1,0]*wd[1,1]+4*(wd[0,0]**3),
                              wd[1,0]**2]
                            ])

    #torch.autograd
    loss_val = L2d(w)
    autograd_grad = gradient_fun(loss_val, w).data
    #emprical gradient
    correct_grad = G2d(w)
    assert (autograd_grad-correct_grad).abs().sum() < TOLERANCE

def test_hessian_fun_1d():
    w = Parameter(torch.rand(3))
    def L(w):
        #A custom loss function
        return (w[0]**2)*w[1]+4*(w[2]**3)*w[0]

    def H(w):
        #Hessian of the L(w)
        wd=w.data
        gw12 = 2*wd[0]
        gw13 = 12*(wd[2]**2)
        gw23 = 0
        gw11 = 2*wd[1]
        gw22 = 0
        gw33 = 24*wd[2]*wd[0]
        return torch.Tensor([[gw11,gw12,gw13],
                             [gw12,gw22,gw23],
                             [gw13,gw23,gw33]])

    a = L(w)
    hessian = hessian_fun(a,w)

    assert torch.sum(torch.abs(hessian-H(w))) < TOLERANCE


def test_hessian_fun_2d():

    w = Parameter(torch.rand(2,2))
    def L2d(w):
        '''
        A custom loss function
            w00 w01
            w10 w11
        '''
        return (w[1,0]**2)*w[1,1]+4*(w[0,0]**3)*w[1,0]

    def H2d(w):
        #Hessian of the L(w)
        wd=w.data
        gw12 = 2*wd[1,0]
        gw13 = 12*(wd[0,0]**2)
        gw23 = 0
        gw11 = 2*wd[1,1]
        gw22 = 0
        gw33 = 24*wd[0,0]*wd[1,0]
        ## x3 0
        ## x1 x2
        return torch.Tensor([[[[gw33,0], #0,0,0,:
                               [gw13,gw23]], #0,0,1,: #x3 with others
                              [[0,0], #0,1,0,:
                               [0,0]]], #0,1,1,:
                             [[[gw13,0], #1,0,0,:
                               [gw11,gw12]], #1,0,1,: #x1 with others
                              [[gw23,0], #1,1,0,:
                              [gw12,gw22]]]]) #1,1,1,: #x2 with others

    a = L2d(w)
    hessian = hessian_fun(a,w)

    assert torch.sum(torch.abs(hessian-H2d(w))) < TOLERANCE

def test_flatten_params():
    from torch import nn
    import torch.nn.functional as F
    from itertools import tee
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
            self.conv2 = nn.Conv2d(2, 1, kernel_size=5)
            self.fc1 = nn.Linear(16, 2)
            self.fc2 = nn.Linear(2, 10)
            self.nonlins = {'conv1':('max_relu',(2,2)),'conv2':('max_relu',(2,2)),'fc1':'relu','fc2':'log_softmax'}

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 16)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x,dim=1)
    model = Net()
    f_params,reverse_gen = flatten_params(model.parameters())

    for a,b in zip(reverse_gen(f_params),model.parameters()):
        assert torch.sum(torch.abs(a-b)).data[0]<TOLERANCE
