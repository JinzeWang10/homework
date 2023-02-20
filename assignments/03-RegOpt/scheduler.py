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
        triangle_len=5000,
        max_lr=0.05,
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
        self.triangle_len = triangle_len
        self.step_add = max_lr / triangle_len
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        get learning rate

        """
        # if self.last_epoch % 1000==0:

        #   print("epoch:{}, avg_lr={}".format(self.last_epoch,np.mean([group["lr"] for group in self.optimizer.param_groups])))
        if self.last_epoch < self.triangle_len:
            return [
                group["lr"] + self.step_add for group in self.optimizer.param_groups
            ]
        elif (
            self.last_epoch >= self.triangle_len
            and self.last_epoch < self.triangle_len * 2
        ):
            return [
                group["lr"] - self.step_add for group in self.optimizer.param_groups
            ]
        else:
            return [
                max(group["lr"] - 0.00001, 0.0003)
                for group in self.optimizer.param_groups
            ]
