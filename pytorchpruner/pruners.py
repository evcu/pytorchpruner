import torch
from torch.nn.modules import Module
from .modules import MaskedModule
from .utils import _find_fan_out_weights,my_select
from torch.nn import Parameter
from torch import nn
from .scorers import magnitudeScorer


def get_pruning_mask(scores,f=0.1):
    """
    @params scores scorer
    @f lowest f fraction of scores elements would have a 1 in the mask.
    """
    srted = torch.sort(scores.view(-1))[0]
    if (len(srted)*f)>(len(srted)-1):
        #In this case we prune all weights
        print("Warning! whole tensor is about to be pruned.")
        thres = float('inf')
    else:
        thres = srted[int(len(srted)*f)]
    mask = scores<thres
    return mask

class BasePruner(object):
    # f_reinit_optimizer needed to be able remove any scheduling existed before,
    # to ensure that the masked gradients are not updated due momentum etc..
    # IDEA maybe it is better to define BasePruner as an optimizer.
    def __init__(self,model,f_reinit_optimizer=lambda : None,scorer=magnitudeScorer):
        if not isinstance(model,MaskedModule):
            if isinstance(model,Module):
                self.masked_model=MaskedModule(model)
            else:
                raise ValueError("model needs to be a nn.Module")
        else:
            self.masked_model = model
        self.scorer = scorer
        self.f_reinit_optimizer = f_reinit_optimizer

    def prune(self,f=0.1):
        #f is prunning factor. TODO make it layer wise possibility. If supplied a list etc..
        # We just prune weights not biases
        # TODO if you want to remove some mask, i.e. f=0.5 after f=0.9. We need to implement that here
        # Don't forget to reinitialize or update the state of the optimizer. Otherwise your weights might get updatete even though they have 0 grad.
        # IDEA maybe use register buffer to hold masks
        if isinstance(f,float):
            for layer,mask in self.masked_model._mask_dict.items():
                scores = self.scorer(layer.weight)
                mask = get_pruning_mask(scores,f)
                layer.weight.data[mask] = 0
                self.masked_model._mask_dict[layer] = mask
        self.f_reinit_optimizer()

    def remove_empty_filters(self, layer_name, nz_frac=0.0, nz_frac_next=0.0, debug=True, out_dim=0, inp_dim=1):
        r"""This function removes dead filters in a conv2d or Linear layer.
        There are two possibilities
        1. All the weights in a filter or row is 0, such that the output always 0
        2. All the weights in the next layer using these weights are 0 such that
            no matter what the filter produces it is not utilized.

        out_dim:  default(1) since input filters are mostly in the first channel

        - Note that a Conv2d layer with 3=inp 16=out channels and 5,5 filter size would have a
        weight tensor of size 16,3,5,5.
        - Also note that a Linear layer of size 256=inp 64=out would have a tensor of size 64*256, so input is in the
        this module

        @params nz_frac: minimum fraction of non_zeros that each units need to satisfy to survive
        @params nz_frac_next: minimum fraction of non_zeros in the outgoing weights for the unit to survive
        """
        layer = getattr(self.masked_model.module,layer_name)
        if isinstance(layer,(nn.Conv2d,nn.Linear)):
            n_filter = layer.weight.data.size(out_dim)
            non_empty_filters = []
            for i in range(n_filter):
                curr_mask = self.masked_model._mask_dict[layer].select(out_dim,i)
                next_layer_name,unit_index = _find_fan_out_weights(self.masked_model.module.defs_conv,
                                                            self.masked_model.module.defs_fc,
                                                            layer_name,i)
                next_layer = getattr(self.masked_model.module,next_layer_name)
                next_mask = my_select(self.masked_model._mask_dict[next_layer],inp_dim,unit_index)
                n_non_zeros = (curr_mask!=0).sum()
                n_total = curr_mask.numel()
                frac_non_zeros = n_non_zeros / float(n_total)
                n_non_zeros_next = (next_mask!=0).sum()
                n_total_next = next_mask.numel()
                frac_non_zeros_next = n_non_zeros_next / float(n_total_next)
                if frac_non_zeros<=nz_frac:
                    if debug: print(f'At i={i}, in_coming: {n_non_zeros}/{n_total} non-zero below {nz_frac}...' )
                elif frac_non_zeros_next<=nz_frac_next:
                    if debug: print(f'At i={i}, out_going: {n_non_zeros_next}/{n_total_next} non-zero below {nz_frac_next}...' )
                else:
                    non_empty_filters.append(i)
            if debug: print('Following filters stays',non_empty_filters)

            if len(non_empty_filters)<n_filter:
                self.remove_filters_(layer,next_layer,non_empty_filters)
            else:
                print('No empty filter found')
        else:
            print('Not implemented')

    def remove_filters_(self,layer,next_layer,surviving_indices,out_dim=0,inp_dim=1):
        new_n_filter = len(surviving_indices)
        # We need to update the layer(weight,bias,mask) and next layer(weigt,mask)
        new_p = Parameter(my_select(layer.weight.data,out_dim,surviving_indices)) #This gets the non-empty filters
        new_p_bias = Parameter(my_select(layer.bias.data,out_dim,surviving_indices)) #This gets the non-empty filters-biases
        new_mask = my_select(self.masked_model._mask_dict[layer],out_dim,surviving_indices) #if you find them select
        if (isinstance(layer,nn.Conv2d)
         and isinstance(next_layer,nn.Linear)):
            # We need to propogate each removed filters output so one filter removed would remove output_size many columns on the next_layer.
            # for output size of 4*4 = 16 we need to have all the 16 indices. So every index in  non_empty_filters saves 16 columns in the Linear layer etc.

            output_size = next_layer.in_features//layer.out_channels
            self.masked_model.module.defs_fc[0] = output_size*new_n_filter ## TODO this is ugly. And requires a defs_fc / defs_conv stuff
            flattened_surviving_columns = [j for k in surviving_indices for j in range(k*output_size,(k+1)*output_size)]
            new_p2 = Parameter(my_select(next_layer.weight.data,inp_dim,flattened_surviving_columns)) ##This gets the non-empty filters-input channels
            layer.out_channels = new_n_filter
            next_layer.in_features = len(flattened_surviving_columns)
            new_mask2 = my_select(self.masked_model._mask_dict[next_layer],inp_dim,flattened_surviving_columns)
        else:
            new_mask2 = my_select(self.masked_model._mask_dict[next_layer],inp_dim,surviving_indices)
            new_p2 = Parameter(my_select(next_layer.weight.data,inp_dim,surviving_indices)) ##This gets the non-empty filters-input channels
            if isinstance(layer,nn.Conv2d):
                layer.out_channels = new_n_filter
            elif isinstance(layer,nn.Linear):
                layer.out_features = new_n_filter
            else:
                raise ValueError('This shouldnt happen: One need to implement setting up necessary layer attributes here')
            if isinstance(next_layer,nn.Conv2d):
                next_layer.in_channels = new_n_filter
            elif isinstance(next_layer,nn.Linear):
                next_layer.in_features = new_n_filter
            else:
                raise ValueError('This shouldnt happen: One need to implement setting up necessary layer attributes here')

        layer.weight = new_p
        layer.bias = new_p_bias
        next_layer.weight = new_p2
        ## Update masks
        self.masked_model._mask_dict[layer] = new_mask
        self.masked_model._mask_dict[next_layer] = new_mask2
