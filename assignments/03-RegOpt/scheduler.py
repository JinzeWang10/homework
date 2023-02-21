from typing import List
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    customize learning rate

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        last_epoch=-1,
        triangle_len=4000,
        max_lr=0.02,
    ) -> None:
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.step_size = step_size
        self.triangle_len = triangle_len
        self.step_add = max_lr * self.step_size / (triangle_len)
        self.step_minus = self.step_add / 2
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        get learning rate

        """
        if self.last_epoch % self.step_size == 0:
            if self.last_epoch < self.triangle_len:
                return [
                    group["lr"] + self.step_add for group in self.optimizer.param_groups
                ]
            elif (
                self.last_epoch >= self.triangle_len
                and self.last_epoch < self.triangle_len * 3
            ):
                return [
                    group["lr"] - self.step_minus
                    for group in self.optimizer.param_groups
                ]
            else:
                return [
                    max(group["lr"] - 0.00001, 0.00001)
                    for group in self.optimizer.param_groups
                ]
        else:
            return [group["lr"] for group in self.optimizer.param_groups]
