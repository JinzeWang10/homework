from typing import List
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.datasets.utils import np


class CustomLRScheduler(_LRScheduler):
    """
    customize learning rate

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float,
        eta_min=0,
        T_max=2,
        last_epoch=-1,
    ) -> None:
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.step_size = step_size
        self.gamma = gamma

        self.eta_min = eta_min
        self.T_max = T_max
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        get learning rate

        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        if self.last_epoch == 0:
            self.lr_base = [group["lr"] for group in self.optimizer.param_groups]
        elif self.last_epoch % self.step_size == 0:
            self.lr_base = [self.gamma * lr for lr in self.lr_base]
        # pi = torch.acos(torch.zeros(1)) * 2
        lr_list = [
            lr * (1 + np.cos(np.pi * self.last_epoch / self.T_max))
            for lr in self.lr_base
        ]
        return lr_list
