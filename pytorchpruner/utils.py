import torch
from torch.autograd import Variable
from torch.nn import Parameter
import collections
from itertools import tee


def index_generator(size):
    #TODO change this with python iterators
    # and maybe multi dim
    if len(size)==1:
        return range(size[0])
    elif len(size)==2:
        return ((i,j) for i in range(size[0]) for j in range(size[1]))
    elif len(size)==3:
        return ((i,j,k) for i in range(size[0]) for j in range(size[1])
                            for k in range(size[2]))
    elif len(size)==4:
        return ((i,j,k,l) for i in range(size[0]) for j in range(size[1])
                            for k in range(size[2]) for l in range(size[3]))
    else:
        raise ValueError("Unsported tensor size: %s, check your code or improve here:index_generator"
                        % str(size))

def generate_onehot_var(size,ind):
    #this to generate one hot vector with size n, x_1=0,x_2=0,...x_i=1,...x_n=0
    res = torch.zeros(size)
    res[ind] = 1
    return Variable(res,requires_grad=False)

def flatten_params(params):
    """
    gets a iterator of Parameter/Variable/Tensor
    returns: [0] returns flatten(1d) version with length N
             [1] a generator function which accepts a Parameter/Variable/Tensor of length N
             and returns a generator of Parameter/Variable/Tensor with same sizes in order as the params.
    """
    if isinstance(params,Parameter):
        return params.contiguous().view(-1)
    else:
        list_of_params = []
        for p in params:
            list_of_params.append(p.contiguous().view(-1))
        return torch.cat(list_of_params)

def get_reverse_flatten_params_fun(params,get_count=False):
    """
    Returns a function which reshapes the flattened vector to its original hessian_shape
    if get_count=True it returs total number of elements for the non-trivial(iterator) case
    """
    if isinstance(params,Parameter):
        def resize_param_fun(flatten_params):
            return flatten_params.view(params.size())
        return resize_param_fun
    else:
        list_of_sizes = []
        def resize_param_fun(flatten_params):
            c_sum = 0
            for numel,size in list_of_sizes:
                yield flatten_params[c_sum:c_sum+numel].view(size)
                c_sum += numel

        if get_count:
            total_element_number = 0
            for p in params:
                total_element_number += p.nelement()
                list_of_sizes.append((p.nelement(),p.size()))
            return resize_param_fun,total_element_number
        else:
            for p in params:
                list_of_sizes.append((p.nelement(),p.size()))
            return resize_param_fun


def hessian_fun(loss,params,flattened=False):
    """
    params: torch.nn.Parameter or iterator of torch.nn.Parameter.
            IMPORTANT if it is an iterator over parameters flattened must be TRUE

            if it is an iterator it returns a list of hessians of each parameter,
            note that each hessian is calculated for the particular Parameter.
            If you want to calculate the hessian of the whole network, please
            flatten the parameter first then call hessian on the flattened parameter.
    returns: torch.<cuda>.Tensor or list of torch.Tensor
        1d of size n->returns n*n tensor.
        2d of size m*n -> returns m*n*m*n tensor.

    NOTE: retains the graph
    TODO:extend to arbitrary params1 params2. such that we can get any part of the big network hessian.
    """
    if isinstance(params,Parameter):
        # Case 1
        pass
    elif isinstance(params,collections.Iterable) and flattened:
        # Case 2
        params = list(params) #this is to prevent grad eating the generator. to use generator multiple times(model.parameters())
        pass
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameters" % type(params))


    jacobian = gradient_fun(loss, params,flattened=flattened, create_graph=True)
    hessian_rows = []
    # Iteratively calculate hessian of L(w) by multipliying the hessian with the one-hot vectors
    # note that ind can be a tuple or single int
    for ind in index_generator(jacobian.size()):
        hessian_row_i = hessian_vector_product( None, #when params_grad given loss is not needed
                                                params,
                                                generate_onehot_var(jacobian.size(),ind),
                                                params_grad = jacobian,
                                                retain_graph = True,
                                                flattened = True)
        hessian_rows.append(hessian_row_i)
    hessian_shape = jacobian.size()*2
    result = torch.stack(hessian_rows).view(hessian_shape)
    return result


def gradient_fun(loss,params,flattened=False,create_graph=False,retain_graph=True):
    """
    loss: a scalar Variable
    params: Parameter with params.size()=S

    returns: Tensor with size S containing the gradient.

    """
    if create_graph:
        #if you are creating it you are reataining it by default.
        retain_graph = True
    gradient = torch.autograd.grad(loss,
                                   params,
                                   create_graph=create_graph,
                                   retain_graph=retain_graph)
    if flattened:
        gradient = flatten_params(gradient)
    elif isinstance(params,Parameter):
        gradient = gradient[0]

    return gradient



def hessian_vector_product(loss,params,vector,params_grad=None,retain_graph=False,flattened=False):
    """
    params: Case 1: Parameter
                Then the param:vector should be a Tensor with same size. The result is same size as the Parameter.
            Case 2: iterator of Parameters
                This is allowed only when flattened=True.
    loss: needed only params_grad is not provided
    vector: Same size as the params_grad. If you are flattened without providing the params_grad note that your vector
            match the size of the flattened parameters.
    params_grad: is for preventing recalculation and to be able to use in hessian
    flattened: if true then the params should be list of parameters. Then the hessian vector product is flattened.
        In this setting I am not returning the reverse functon that flatten_params generate since
        the only instance where I flatten is during the hessian and I get the same function during grad calcualtion.
        Future use cases may require and one can return.
    """

    #We need a Variable, so ensure
    if torch.is_tensor(vector):
        vector = Variable(vector,requires_grad=False)
    elif isinstance(vector,Variable):
        pass
    else:
        raise ValueError("Vector passed is not a Variable or Tensor: {}".format(type(vector)))

    if isinstance(params,Parameter):
        # Case 1
        pass
    elif isinstance(params,collections.Iterable) and flattened:
        # Case 2
        params = list(params)
        pass
    else:
        raise ValueError("Invalid type, received: %s. either supply iterable of \
                            parameters or a single parameters" % type(params))

    if isinstance(params_grad,Variable):
        pass
    else:
        params_grad = torch.autograd.grad(loss, params, create_graph=True)
        if flattened:
            params_grad = flatten_params(params_grad)
        else:
            params_grad = params_grad[0]
    if params_grad.is_cuda: vector= vector.cuda()
    # import pdb;pdb.set_trace()
    grad_vector_dot = torch.sum(params_grad * vector)
    hv_params = torch.autograd.grad(grad_vector_dot, params,retain_graph=retain_graph)
    if flattened:
        hv_params = flatten_params(hv_params)
    else:
        hv_params = hv_params[0]

    return hv_params.data

def my_select(tensor,dim,indices):
    if dim==0:
        return tensor[indices,]
    elif dim==1:
        return tensor[:,indices,]
    elif dim==2:
        return tensor[:,:,indices,]
    else:
        raise ValueError('That is enough')

def _find_fan_out_weights(defs_conv,defs_fc,layer_name,unit_index):
    """
    Arguments:
        - unit_index: int or slice
    finding outgoing layer and the indices associated with the unit provided.
    1. Find layer, check unit_index
    2. Save consecutive layer
    3. If the layer=conv next_layer=fc we need to update the slice
    Example:

        self.defs_conv = [1,
                        ('conv1',8,5,2),
                        ('conv2',16,5,2)]
        self.defs_fc = [16*16,
                        ('fc1',64),
                        ('fc2',10)]
    """
    is_found = False
    next_layer = False

    #if slice we need better error checking
    if isinstance(unit_index,slice):
        compare_f = lambda i,n_out: 0<=min(i.start,i.stop) and max(i.start,i.stop)<n_out
    else:
        compare_f = lambda i,n_out: 0<=i<n_out

    for l,n_out,_,_ in defs_conv[1:]:
        if is_found:
            next_layer = l
            break
        if layer_name==l:
            is_found = True
            #check index is valid
            if not compare_f(unit_index,n_out):
                raise ValueError(f'index:{unit_index} is not 0<=x<{n_out} at layer: {layer_name}')
    if not next_layer:
        if is_found:
            #This means the last conv layer is the layer and we need a special handling
            #since there are might be multiple weights associated with theses outputs
            next_layer = defs_fc[1][0]
            conv_out_numel = defs_fc[0]//n_out
            #we need to expand the slice
            if isinstance(unit_index,slice):
                unit_index = slice(unit_index.start*conv_out_numel,unit_index.stop*conv_out_numel)
            else:
                start_index = unit_index*conv_out_numel
                unit_index = slice(start_index,start_index+conv_out_numel)
        else:
            for l,n_out in defs_fc[1:]:
                if is_found:
                    next_layer = l
                    break
                if layer_name==l:
                    is_found = True
                    if not compare_f(unit_index,n_out):
                        raise ValueError(f'index:{unit_index} is not 0<=x<{n_out} at layer: {layer_name}')
    return next_layer,unit_index
