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
        self.layer2 = nn.Linear(hidden_size, 64)
        self.layer3 = nn.Linear(64, 32)
        initializer(self.layer1.weight)
        initializer(self.layer2.weight)
        initializer(self.layer3.weight)
        # self.layers.append()

        self.out = nn.Linear(32, num_classes)
        initializer(self.out.weight)
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
        x = self.activation(self.layer3(x))
        # for layer in self.layers:
        #   x = self.activation(layer(x))
        # Get outputs
        x = self.out(x)

        return x


# COnclusion: (1) ones_init always produce best result;
# (2) larger hidden dim creates better result
# (3)

# MLP(input_dim, 64, output_dim, 1, torch.nn.functional.sigmoid, torch.nn.init.xavier_normal_)
# Epoch: 9, Accuracy: 0.9678: 100%|██████████| 10/10 [01:40<00:00, 10.08s/it]

# MLP(input_dim, 10, output_dim, 1, torch.nn.functional.sigmoid, torch.nn.init.xavier_normal_)
# Epoch: 9, Accuracy: 0.9188: 100%|██████████| 10/10 [01:33<00:00,  9.33s/it]

# MLP(input_dim, 128, output_dim, 1, torch.nn.functional.sigmoid, torch.nn.init.xavier_normal_)
# Epoch: 9, Accuracy: 0.9759: 100%|██████████| 10/10 [01:51<00:00, 11.12s/it]


# MLP(input_dim, 128, output_dim, 1, torch.nn.functional.sigmoid, torch.nn.init.uniform_)
# Epoch: 9, Accuracy: 0.1028: 100%|██████████| 10/10 [01:54<00:00, 11.50s/it]

# MLP(input_dim, 128, output_dim, 1, torch.nn.functional.softmax, torch.nn.init.uniform_)
# Epoch: 9, Accuracy: 0.9437: 100%|██████████| 10/10 [01:58<00:00, 11.89s/it]

# MLP(input_dim, 64, output_dim, 1, torch.nn.functional.hardtanh, torch.nn.init.uniform_)


# MLP(input_dim, 64, output_dim, 1, torch.nn.functional.hardtanh, torch.nn.init.normal_)


# MLP(input_dim, 64, output_dim, 1, torch.nn.functional.hardtanh, torch.nn.init.xavier_normal_)
# Epoch: 9, Accuracy: 0.9717: 100%|██████████| 10/10 [01:43<00:00, 10.31s/it]

# MLP(input_dim, 32, output_dim, 1, torch.nn.functional.sigmoid, torch.nn.init.xavier_normal_)


# MLP(input_dim, 10, output_dim, 1, torch.nn.functional.sigmoid, torch.nn.init.ones_)
# Epoch: 9, Accuracy: 0.1135: 100%|██████████| 10/10 [01:34<00:00,  9.42s/it]

# MLP(input_dim, 10, output_dim, 1, torch.nn.functional.hardtanh, torch.nn.init.ones_)
# Epoch: 9, Accuracy: 0.1135: 100%|██████████| 10/10 [01:34<00:00,  9.43s/it]

# MLP(input_dim, 32, output_dim, 1, torch.nn.functional.hardtanh, torch.nn.init.ones_)
# Epoch: 9, Accuracy: 0.1135: 100%|██████████| 10/10 [01:37<00:00,  9.71s/it]

# MLP(input_dim, 64, output_dim, 1, torch.nn.functional.hardtanh, torch.nn.init.ones_)
# Epoch: 9, Accuracy: 0.1135: 100%|██████████| 10/10 [01:42<00:00, 10.23s/it]


# actv:sigmoid,softmax,glu,rrelu,elu,hardswish,relu,threshold
