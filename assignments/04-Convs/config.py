from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
    batch_size = 16
    num_epochs = 4

    initial_learning_rate = 0.001
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler constructor here.
        "step_size": 4,
        "triangle_len": 800,
        "max_lr": 0.005,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )
    # optimizer_factory: Callable[
    #     [nn.Module], torch.optim.Optimizer
    # ] = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3)

    transforms = Compose([ToTensor()])
