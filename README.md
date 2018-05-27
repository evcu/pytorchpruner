# pytorchpruner

Do `python setup.py install` to install.
Check `notebooks/` for samples.

`pytorchpruner` is a [pytorch](https://pytorch.org/) package for pruning neural networks. It is intended for research and its main objective is not to provide fastest pruning framework, however it is relatively efficient and fast. It uses masking idea to simulate pruning and supports two main pruning strategies. It also implements various second order functions like hessian and hessian-vector product.

There are X main parts of the library

1. **Parameter Pruning (pytorchpruner.scorers)**: Saliency measures that return a same-sized-tensor of scores for each parameter in the provided parameter tensor. 
2. **Unit Pruning (pytorchpruner.unitscorers)**: Saliency measures that return a vector of scores for each unit in the provided parameter tensor.
3. **Pruners (pytorchpruner.pruners)**: Has two different pruner engine for the two different pruning strategies (parameter vs unit). `remove_empty_filters` function in this file reduces the size of the network by copying the parameters into smaller tensors if possible.
4. **Auxiliary Modules (pytorchpruner.modules)**: implements `meanOutputReplacer` and `maskedModule`, the two important wrapper for `torch.nn.Module` instances. The first one replaces its output with the mean value, if enabled. And the second one simulates the pruning layers.
5. **Various first/second-order functionality (pytorchpruner.utils)**: implements hessian calculation, hessian-vector product, search functionality and some other utility functions.
