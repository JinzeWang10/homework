import torch


class Model(torch.nn.Module):
    """
    model
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        init the model
        """
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channels, 28, 3, 3)
        # nn.init.dirac_(self.conv1.weight)
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=28)
        # nn.init.xavier_uniform(self.batch_norm1.weight)

        self.fc1 = torch.nn.Linear(700, 128)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = torch.nn.Linear(128, num_classes)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward layer
        """
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.batch_norm1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)

        x = torch.nn.functional.relu(self.fc2(x))
        # print(x.shape)
        return x
