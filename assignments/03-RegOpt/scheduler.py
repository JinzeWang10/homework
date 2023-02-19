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
        if self.last_epoch / 782 <= 10:
            if self.last_epoch == 0 or (self.last_epoch % self.step_size) != 0:
                return [group["lr"] for group in self.optimizer.param_groups]
            else:
                pi = torch.acos(torch.zeros(1)) * 2
                lr = [
                    self.gamma
                    * (
                        self.eta_min
                        + 0.5
                        * abs(group["lr"] - self.eta_min)
                        * (1 + torch.cos(pi * self.last_epoch / self.T_max))
                    )
                    for group in self.optimizer.param_groups
                ]
                return [i.item() for i in lr]
        else:
            pi = torch.acos(torch.zeros(1)) * 2
            lr = [
                0.9
                * (
                    self.eta_min
                    + 0.5
                    * abs(group["lr"] - self.eta_min)
                    * (1 + torch.cos(pi * self.last_epoch / self.T_max))
                )
                for group in self.optimizer.param_groups
            ]
            return [i.item() for i in lr]
