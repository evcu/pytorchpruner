import torch
from torch.nn.modules import Module
from .modules import MaskedModule
from torch.nn import Parameter
from torch import nn

class BasePruner(object):
    # One can add trainer here.
    def __init__(self,model):
        if not isinstance(model,MaskedModule):
            if isinstance(model,Module):
                self.masked_model=MaskedModule(model)
            else:
                raise ValueError("model needs to be a nn.Module")
        else:
            self.masked_model = model

    def prune(self,f=0.1):
        #f is prunning factor. TODO make it layer wise possibility. If supplied a list etc..
        # We just prune weights not biases
        # this function can get flag like score_function='magnitude' and this would trigger appropriate __scorefunction.
        # TODO if you want to remove some mask, i.e. f=0.5 after f=0.9. We need to implement that here
        # Don't forget to reinitialize or update the state of the optimizer. Otherwise your weights might get updatete eve though they have 0 grad.
        # IDEA maybe use register buffer to hold masks
        if isinstance(f,float):
            for layer,mask in self.masked_model._mask_dict.items():
                scores = self.score_function(layer)
                srted = torch.sort(scores.view(-1))[0]
                if (len(srted)*f)>(len(srted)-1):
                    #In this case we prune all weights
                    print("Warning! whole layer is about to be pruned.")
                    thres = float('inf')
                else:
                    thres = srted[int(len(srted)*f)]
                mask = scores>=thres
                layer.weight.data[mask!=1] = 0
                self.masked_model._mask_dict[layer] = mask
    def remove_empty_filters(self, layer, next_layer, nz_threshold=0, nz_threshold_next=0, debug=True, out_dim=0, inp_dim=1):
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
        """
        if (isinstance(layer,(nn.Conv2d,nn.Linear))
         and isinstance(next_layer,(nn.Conv2d,nn.Linear))):
            n_filter = layer.weight.data.size(out_dim)
            non_empty_filters = []
            for i in range(n_filter):
                curr_mask = self.masked_model._mask_dict[layer].select(out_dim,i)
                next_mask = self.masked_model._mask_dict[next_layer].select(inp_dim,i)
                n_non_zeros = (curr_mask!=0).sum()
                n_non_zeros_next = (next_mask!=0).sum()
                if n_non_zeros<=nz_threshold:
                    if debug: print('At i=%d, Weights: %d non-zero...' % (i,n_non_zeros))
                elif n_non_zeros_next<=nz_threshold_next:
                    if debug: print('At i=%d, Next_weights: %d non-zero...' % (i,n_non_zeros_next))
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
        def my_select(tensor,dim,indices):
            if dim==0:
                return tensor[indices,]
            elif dim==1:
                return tensor[:,indices,]
            elif dim==2:
                return tensor[:,:,indices,]
            else:
                raise ValueError('That is enough')
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

    def score_function(self,layer):
        #must return non-negative scores. High scores represents high saliency.
        return layer.weight.data.clone().abs()
