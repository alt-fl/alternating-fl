from torch.nn import Module

from .lenet5 import LeNet5
from .cnn import CNN


def get_model(name: str, **kwargs) -> Module:
    """
    name: the name of the model
    """
    match name.lower():
        case "lenet5":
            model = LeNet5(name, **kwargs)
        case "cnn":
            model = CNN(name, **kwargs)
        case _:
            raise ValueError(f"the model {name!r} is not supported")

    return model
