import inspect
import warnings

import torch.nn as nn

# Dictionary to map names to activation modules
activation_map = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "log_softmax": nn.LogSoftmax,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "prelu": nn.PReLU,
    "hardswish": nn.Hardswish,
    "hardsigmoid": nn.Hardsigmoid,
}


# function to get the activation module by name
def get_activation(name, **kwargs):
    # Select the activation class
    activation_class = activation_map.get(name.lower())
    if activation_class is None:
        raise ValueError(f"Unsupported activation function: {name}")

    # filter kwargs to only those supported by the activation's __init__ method
    sig = inspect.signature(activation_class)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if kwargs != valid_kwargs:
        warnings.warn(
            f"Unsupported kwargs for activation {name}: {set(kwargs) - set(valid_kwargs)}"
        )

    return activation_class(**valid_kwargs)
