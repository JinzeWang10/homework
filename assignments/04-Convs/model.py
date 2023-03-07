import torch


class Model(torch.nn.Module):
    """
    Try to train a fast CNN model
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        init the model
        """
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channels, 16, 3, 3)
        # self.conv2 = torch.nn.Conv2d(16, 4, 1, 1)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=16)
        # self.batch_norm2 = torch.nn.BatchNorm2d(num_features=4)
        # self.maxpool1=nn.MaxPool2d(13,1)
        self.fc1 = torch.nn.Linear(400, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward layer
        """
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_norm1(x)
        # x = self.conv2(x)
        # x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        # x = self.batch_norm2(x)
        x = torch.nn.functional.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)
        # print(x.shape)
        return x
