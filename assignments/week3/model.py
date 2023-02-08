import torch
import torch.nn as nn
from typing import Callable


class MLP(nn.Module):
    """
    Initialize the MLP.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.activation = activation

        # self.layers=nn.ModuleList()
        # self.layers.append()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 32)
        # self.layer3 = nn.Linear(64, 32)
        initializer(self.layer1.weight)
        initializer(self.layer2.weight)
        # initializer(self.layer3.weight)
        # self.layers.append()

        self.out = nn.Linear(32, num_classes)
        initializer(self.out.weight)

        # self.layer=nn.Linear(input_size,num_classes)
        # initializer(self.out.weight)
        # self.layers.append()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """

        # Get activations of each layer
        # print(x)
        x = self.activation(self.layer1(x))
        # print(x)
        x = self.activation(self.layer2(x))
        # x = self.activation(self.layer3(x))

        # Get outputs

        x = self.out(x)

        return self.activation(x)


# COnclusion: (1) ones_init always produce best result;
# (2) larger hidden dim creates better result
# (3)

# 3 lyaers 128 64 32 relu--------0.09
# 2 layers 128 64 relu ----------
# 2 layers 128 64 softmax ----------


# actv:sigmoid,softmax,glu,rrelu,elu,hardswish,relu,threshold
