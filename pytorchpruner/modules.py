import torch
from torch.nn.modules import Module
def norm_i(tensr):
    n_units = tensr.size(1)
    norm_arr = torch.zeros(n_units)
    for ui in range(n_units):
        norm_arr[ui] = tensr[:,ui,].norm(p=1)
    return norm_arr.cpu()

class meanOutputReplacer(torch.nn.Module):
    def __init__(self,module,unit_id=0):
        """
        meanOutputReplacer warps any module single output module.
        This module has two purpose(mode) and they can be switched with the flags 'enabled' and 'is_mean_replace'
        @params enabled Enables the module, if not enabled the forward is equal to the result of the wrapped Module
        @params is_mean_replace if the flag `enabled` is also true, the output `unit_id`th unit replaced with its mean.
            if not true: than only the zero_mean output is calculated, to be able to calculate the MRS score efficiently
            on the back-pass-hook.
        """
        super(meanOutputReplacer, self).__init__()
        if not isinstance(unit_id,int):
            raise ValueError('not implemented must be int')
            ## note that implementing for whole layer is straight forward, just define bunch of 1's
        if isinstance(module,meanOutputReplacer):
            raise ValueError("ERROR: module given is a meanOutputReplacer, this should be wrong")
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.enabled = False
        self.is_mean_replace = False
        self.unit_id = unit_id
        def backwardhook(l,inp,out):
            if l.enabled and not l.is_mean_replace:
                prdct = (out[0].data*l.cy_zeromean)
                self.mrss = norm_i(prdct)
        self.register_backward_hook(backwardhook)

    def forward(self,*inputs, **kwargs):
        if len(inputs)>1:
            raise ValueError('meanOutputReplacer is not implemented for layyer getting multiple inputs')
        if self.enabled:
            out = self.module(*inputs, **kwargs)

            out_mean = out.mean(0)
            # second dim is the n_outputs
            while out_mean.dim()>1:
                out_mean = out_mean.mean(1)
            self.cy_mean = out_mean.data
            # import pdb;pdb.set_trace()
            if self.is_mean_replace:
                out[:,self.unit_id] = out_mean[self.unit_id].expand(out.size(0),*out.size()[2:])
            else:
                out_mean_expanded = out_mean.data.expand(out.size(0),
                                                        *out.size()[2:],
                                                        -1)
                if out_mean_expanded.dim()>2:
                    out_mean_expanded= out_mean_expanded.transpose(1,3)
                self.cy_zeromean = out.data-out_mean_expanded
        else:
            out = self.module(*inputs, **kwargs)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(\n\t' \
            + 'module=' + str(self.module) \
            + '\n\t,is_mean_replace=' + str(self.is_mean_replace) \
            + '\n\t,enabled=' + str(self.enabled) + ')'

class MaskedModule(Module):
    r"""Implements masked module for prunning etc...
    it creates a mask for each layer and holds those masks inside a dictionary belongs to
    this module
    """
    DEFAULT_MASKED_MODULES = (torch.nn.Conv2d,torch.nn.Linear)
    def __init__(self, module, masked_modules=DEFAULT_MASKED_MODULES):
        super().__init__()
        self.module = module
        self.def_masked_modules = masked_modules
        self._mask_dict = {} # nn.Module->torch.ByteTensor
        # self._inp_dict = {} # nn.Module->torch.Tensor
        # self._ginp_dict = {} # nn.Module->torch.Tensor
        # self.__fhook_dict = {} # nn.Module->fuction
        # self.__bhook_dict = {} # nn.Module->fuction
        self.initialize_transparent_masks()

    def initialize_transparent_masks(self):
        def helper(m):
            if isinstance(m, self.def_masked_modules):
                self._mask_dict[m] = [torch.zeros(m.weight.data.size()).byte(),
                                    torch.zeros(m.bias.data.size()).byte()]
        self.module.apply(helper)

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def cpu(self):
        for k,v in self._mask_dict.items():
            self._mask_dict[k]=[v[0].cpu(),v[1].cpu()]
        self.module.cpu()
        super().cpu()
        return self
    def cuda(self):
        for k,v in self._mask_dict.items():
            self._mask_dict[k]=[v[0].cuda(),v[1].cuda()]
        self.module.cuda()
        super().cuda()
        return self
    def apply_mask_on_gradients(self):
        def helper(m):
            if isinstance(m, self.def_masked_modules):
                m.weight.grad.data[self._mask_dict[m][0]]=0
                m.bias.grad.data[self._mask_dict[m][1]]=0
        self.module.apply(helper)

    def calculateSparsity(self):
        sum_zeros = 0
        sum_elements = 0
        if self._mask_dict:
            for mask_w,mask_b in self._mask_dict.values():
                sum_zeros += mask_w.sum()+mask_b.sum()
                sum_elements += mask_w.nelement() + mask_b.nelement()
            return (sum_zeros/sum_elements)
        else:
            error('Mask is not initilized, this should not happen currently. 01.04.2018')
    def __repr__(self):
        return self.__class__.__name__ + '(\n\t' \
            + 'module=' + str(self.module) + ')'
    # def initiliaze_forward_hooks(self):
    #     def f_hook(module,inp,out):
    #         #TODO check inp is a tuple of size 1.
    #         self._inp_dict[module]=inp[0]
    #     def helper(m):
    #         if isinstance(m, self.DEFAULT_MASKED_MODULES):
    #             self.__fhook_dict[m] = m.register_forward_hook(f_hook)
    #     self.module.apply(helper)
    #
    # def remove_forward_hooks(self):
    #     if self.__fhook_dict:
    #         for v in self.__fhook_dict.values():
    #             v.remove()
    #         self._inp_dict = {}
    #         self.__fhook_dict = {}
    #
    # def initiliaze_backward_hooks(self):
    #     def b_hook(module,ginp,gout):
    #         #TODO check inp is Tensor or tuple
    #         self._ginp_dict[module]=ginp
    #     def helper(m):
    #         if isinstance(m, self.DEFAULT_MASKED_MODULES):
    #             self.__bhook_dict[m] = m.register_backward_hook(b_hook)
    #     self.module.apply(helper)
    #
    # def remove_backward_hooks(self):
    #     if self.__bhook_dict:
    #         for v in self.__bhook_dict.values():
    #             v.remove()
    #         self._ginp_dict = {}
    #         self.__bhook_dict = {}
