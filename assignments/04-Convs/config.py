from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
<<<<<<< HEAD
    batch_size = 55
    num_epochs = 4

    initial_learning_rate = 0.001
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler constructor here.
        "step_size": 4,
        "triangle_len": 800,
        "max_lr": 0.005,
    }
=======
    batch_size = 48
    num_epochs = 8
>>>>>>> 740ae9e951de7f55f83c82b908d8f95754404a75

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=0.00125)

    transforms = Compose([ToTensor()])
