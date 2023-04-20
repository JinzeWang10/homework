import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class Net(nn.Module):
    """
    Disallow missing docstrings.

    """

    def __init__(self, states_dims: int, n_actions: int) -> None:
        """
        Disallow missing docstrings.

        """
        super().__init__()
        self.fc1 = nn.Linear(in_features=states_dims, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Disallow missing docstrings.

        """
        x = torch.from_numpy(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x


class Agent:
    """
    Disallow missing docstrings.

    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """
        Disallow missing docstrings.

        """
        self.epsilon = 0.3
        self.batch_size = 64
        self.epsilon_decay = 0.001
        self.epsilon_final = 0.01
        self.lr = 0.00075
        self.discount = 0.9
        self.action_space = action_space
        self.observation_space = observation_space
        self.update_rate = 128

        self.net = Net(8, 4)
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Disallow missing docstrings.

        """

        if np.random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                actions = self.net(torch.from_numpy(observation))
            action = np.argmax(actions)

        self.state = observation
        return action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
    ) -> None:
        """
        Disallow missing docstrings.

        """
        q_pred = self.net(self.state).max(dim=1)
        q_next_max = torch.max(self.net(observation), 1)
        q_target = reward + self.discount * q_next_max * (1 - terminated)

        L = nn.CrossEntropyLoss()
        loss = L(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return
