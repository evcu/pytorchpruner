import torch
from torch.nn.modules import Module

class MaskedModule(Module):
    r"""Implements masked module for prunning etc...
    it creates a mask for each layer and holds those masks inside a dictionary belongs to
    this module
    """
    DEFAULT_MASKED_MODULES = (torch.nn.Conv2d,torch.nn.Linear)
    def __init__(self, module, masked_modules=DEFAULT_MASKED_MODULES):
        super(MaskedModule, self).__init__()
        self.module = module
        self._mask_dict = {} # nn.Module->torch.ByteTensor
        self._inp_dict = {} # nn.Module->torch.Tensor
        self._ginp_dict = {} # nn.Module->torch.Tensor
        self.__fhook_dict = {} # nn.Module->fuction
        self.__bhook_dict = {} # nn.Module->fuction
        self.initialize_transparent_masks()

    def initialize_transparent_masks(self):
        def helper(m):
            if isinstance(m, self.DEFAULT_MASKED_MODULES):
                self._mask_dict[m] = torch.zeros(m.weight.data.size()).byte()
        self.module.apply(helper)

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def apply_mask_on_gradients(self):
        def helper(m):
            if isinstance(m, self.DEFAULT_MASKED_MODULES):
                m.weight.grad.data[self._mask_dict]=0
        self.module.apply(helper)

    def initiliaze_forward_hooks(self):
        def f_hook(module,inp,out):
            #TODO check inp is a tuple of size 1.
            self._inp_dict[module]=inp[0]
        def helper(m):
            if isinstance(m, self.DEFAULT_MASKED_MODULES):
                self.__fhook_dict[m] = m.register_forward_hook(f_hook)
        self.module.apply(helper)

    def remove_forward_hooks(self):
        if self.__fhook_dict:
            for v in self.__fhook_dict.values():
                v.remove()
            self._inp_dict = {}
            self.__fhook_dict = {}

    def initiliaze_backward_hooks(self):
        def b_hook(module,ginp,gout):
            #TODO check inp is Tensor or tuple
            self._ginp_dict[module]=ginp
        def helper(m):
            if isinstance(m, self.DEFAULT_MASKED_MODULES):
                self.__bhook_dict[m] = m.register_backward_hook(b_hook)
        self.module.apply(helper)

    def remove_backward_hooks(self):
        if self.__bhook_dict:
            for v in self.__bhook_dict.values():
                v.remove()
            self._ginp_dict = {}
            self.__bhook_dict = {}

    def calculateSparsity(self):
        sum_zeros = 0
        sum_elements = 0
        if self._mask_dict:
            for mask in self._mask_dict.values():
                sum_zeros += mask.sum()
                sum_elements += mask.nelement()
            return (sum_zeros/sum_elements)
        else:
            error('Mask is not initilized, this should not happen currently. 01.04.2018')
    def __repr__(self):
        return self.__class__.__name__ + '(\n\t' \
            + 'module=' + str(self.module) + ')'
