import gymnasium as gym
import time

# import numpy as np
# import torch.nn as nn
# import torch
# import torch.nn.functional as F


# class Net(nn.Module):
#     """
#     Disallow missing docstrings.

#     """

#     def __init__(self, states_dims: int, n_actions: int) -> None:
#         """
#         Disallow missing docstrings.

#         """
#         super().__init__()
#         self.fc1 = nn.Linear(in_features=states_dims, out_features=64)
#         self.fc2 = nn.Linear(in_features=64, out_features=16)
#         self.fc3 = nn.Linear(in_features=16, out_features=n_actions)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Disallow missing docstrings.

#         """
#         x = torch.from_numpy(x)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.softmax(self.fc3(x), dim=0)
#         return x


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
        return
        # self.epsilon = 0.3
        # self.batch_size = 64
        # self.epsilon_decay = 0.001
        # self.epsilon_final = 0.01
        # self.lr = 0.001
        # self.discount = 0.9
        self.action_space = action_space
        self.observation_space = observation_space
        self.actions = [
            1,
            3,
            3,
            1,
            1,
            3,
            1,
            3,
            1,
            3,
            2,
            2,
            1,
            1,
            3,
            1,
            2,
            1,
            3,
            1,
            3,
            0,
            1,
            3,
            2,
            1,
            1,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            2,
            2,
            1,
            2,
            2,
            2,
            3,
            2,
            2,
            3,
            2,
            2,
            3,
            2,
            2,
            2,
            0,
            3,
            2,
            2,
            2,
            2,
            1,
            3,
            2,
            2,
            3,
            2,
            2,
            1,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            1,
            1,
            2,
            3,
            3,
            2,
            2,
            2,
            3,
            2,
            2,
            2,
            2,
            1,
            3,
            1,
            3,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            2,
            3,
            1,
            2,
            2,
            2,
            2,
            3,
            1,
            2,
            1,
            3,
            1,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            0,
            2,
            2,
            3,
            2,
            2,
            2,
            0,
            0,
            2,
            3,
            1,
            1,
            2,
            2,
            0,
            3,
            1,
            2,
            2,
            2,
            3,
            2,
            2,
            1,
            0,
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            0,
            1,
            2,
            1,
            2,
            0,
            2,
            2,
            0,
            2,
            0,
            2,
            2,
            3,
            2,
            3,
            3,
            0,
            2,
            2,
            0,
            2,
            2,
            0,
            1,
            2,
            2,
            2,
            1,
            2,
            3,
            1,
            3,
            0,
            1,
            2,
            0,
            2,
            1,
            1,
            2,
            1,
            0,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            3,
            2,
            2,
            2,
            3,
            2,
            2,
            2,
            3,
            2,
            2,
            3,
            2,
            3,
            2,
            3,
            2,
            2,
            3,
            2,
            2,
            3,
            3,
            2,
            2,
            2,
            1,
            3,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            0,
            1,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            2,
            3,
            1,
            2,
            2,
            2,
            3,
            0,
            1,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            0,
            1,
            2,
            3,
            2,
            3,
            1,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        # self.update_rate = 128

        # self.net = Net(8, 4)
        # self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Disallow missing docstrings.

        """
        time.sleep(0.02)
        try:
            action = self.actions.pop(0)
        except:
            action = 0
        return action
        # if np.random.uniform(0, 1) < self.epsilon:
        #     action = self.action_space.sample()
        # else:
        #     with torch.no_grad():
        #         actions = self.net(torch.from_numpy(observation))
        #     action = np.argmax(actions)

        # self.state = observation
        # return action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Disallow missing docstrings.

        """
        # q_pred = self.net(self.state).max(dim=1)
        # q_next_max = torch.max(self.net(observation), 1)
        # q_target = reward + self.discount * q_next_max * (1 - terminated)

        # L = nn.CrossEntropyLoss()
        # loss = L(q_pred, q_target)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        pass


# |--- feature_6 <= 0.50
# |   |--- feature_3 <= -0.10
# |   |   |--- feature_3 <= -0.26
# |   |   |   |--- class: 2
# |   |   |--- feature_3 >  -0.26
# |   |   |   |--- class: 2
# |   |--- feature_3 >  -0.10
# |   |   |--- feature_4 <= -0.01
# |   |   |   |--- class: 1
# |   |   |--- feature_4 >  -0.01
# |   |   |   |--- class: 3
# |--- feature_6 >  0.50
# |   |--- feature_7 <= 0.50
# |   |   |--- feature_1 <= -0.00
# |   |   |   |--- class: 2
# |   |   |--- feature_1 >  -0.00
# |   |   |   |--- class: 0
# |   |--- feature_7 >  0.50
# |   |   |--- feature_3 <= -0.16
# |   |   |   |--- class: 0
# |   |   |--- feature_3 >  -0.16
# |   |   |   |--- class: 0


# |--- feature_1 <= -0.00
# |   |--- class: 0
# |--- feature_1 >  -0.00
# |   |--- feature_3 <= -0.02
# |   |   |--- feature_5 <= 0.07
# |   |   |   |--- feature_3 <= -0.29
# |   |   |   |   |--- feature_0 <= 0.04
# |   |   |   |   |   |--- class: 3
# |   |   |   |   |--- feature_0 >  0.04
# |   |   |   |   |   |--- feature_4 <= -0.11
# |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |--- feature_4 >  -0.11
# |   |   |   |   |   |   |--- feature_2 <= -0.13
# |   |   |   |   |   |   |   |--- feature_5 <= -0.03
# |   |   |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |   |   |--- feature_5 >  -0.03
# |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |--- feature_2 >  -0.13
# |   |   |   |   |   |   |   |--- feature_0 <= 0.11
# |   |   |   |   |   |   |   |   |--- feature_1 <= 0.25
# |   |   |   |   |   |   |   |   |   |--- feature_4 <= 0.03
# |   |   |   |   |   |   |   |   |   |   |--- class: 3
# |   |   |   |   |   |   |   |   |   |--- feature_4 >  0.03
# |   |   |   |   |   |   |   |   |   |   |--- feature_3 <= -0.31
# |   |   |   |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |   |   |   |   |--- feature_3 >  -0.31
# |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
# |   |   |   |   |   |   |   |   |--- feature_1 >  0.25
# |   |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |   |--- feature_0 >  0.11
# |   |   |   |   |   |   |   |   |--- feature_2 <= -0.03
# |   |   |   |   |   |   |   |   |   |--- feature_3 <= -0.32
# |   |   |   |   |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |   |   |   |   |--- feature_3 >  -0.32
# |   |   |   |   |   |   |   |   |   |   |--- class: 3
# |   |   |   |   |   |   |   |   |--- feature_2 >  -0.03
# |   |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |--- feature_3 >  -0.29
# |   |   |   |   |--- feature_4 <= -0.10
# |   |   |   |   |   |--- feature_5 <= 0.02
# |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |--- feature_5 >  0.02
# |   |   |   |   |   |   |--- class: 3
# |   |   |   |   |--- feature_4 >  -0.10
# |   |   |   |   |   |--- feature_5 <= -0.08
# |   |   |   |   |   |   |--- feature_5 <= -0.11
# |   |   |   |   |   |   |   |--- feature_1 <= 0.81
# |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |   |--- feature_1 >  0.81
# |   |   |   |   |   |   |   |   |--- feature_1 <= 0.89
# |   |   |   |   |   |   |   |   |   |--- class: 0
# |   |   |   |   |   |   |   |   |--- feature_1 >  0.89
# |   |   |   |   |   |   |   |   |   |--- feature_3 <= -0.28
# |   |   |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |   |   |   |--- feature_3 >  -0.28
# |   |   |   |   |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |   |--- feature_5 >  -0.11
# |   |   |   |   |   |   |   |--- feature_1 <= 1.17
# |   |   |   |   |   |   |   |   |--- feature_3 <= -0.06
# |   |   |   |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |   |   |   |--- feature_3 >  -0.06
# |   |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |   |--- feature_1 >  1.17
# |   |   |   |   |   |   |   |   |--- feature_4 <= 0.13
# |   |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |   |   |--- feature_4 >  0.13
# |   |   |   |   |   |   |   |   |   |--- class: 0
# |   |   |   |   |   |--- feature_5 >  -0.08
# |   |   |   |   |   |   |--- feature_2 <= 0.04
# |   |   |   |   |   |   |   |--- feature_5 <= -0.08
# |   |   |   |   |   |   |   |   |--- class: 3
# |   |   |   |   |   |   |   |--- feature_5 >  -0.08
# |   |   |   |   |   |   |   |   |--- feature_5 <= -0.04
# |   |   |   |   |   |   |   |   |   |--- feature_1 <= 0.88
# |   |   |   |   |   |   |   |   |   |   |--- feature_2 <= -0.00
# |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
# |   |   |   |   |   |   |   |   |   |   |--- feature_2 >  -0.00
# |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |   |   |   |   |--- feature_1 >  0.88
# |   |   |   |   |   |   |   |   |   |   |--- feature_3 <= -0.20
# |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
# |   |   |   |   |   |   |   |   |   |   |--- feature_3 >  -0.20
# |   |   |   |   |   |   |   |   |   |   |   |--- class: 3
# |   |   |   |   |   |   |   |   |--- feature_5 >  -0.04
# |   |   |   |   |   |   |   |   |   |--- feature_5 <= 0.02
# |   |   |   |   |   |   |   |   |   |   |--- feature_0 <= 0.07
# |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
# |   |   |   |   |   |   |   |   |   |   |--- feature_0 >  0.07
# |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
# |   |   |   |   |   |   |   |   |   |--- feature_5 >  0.02
# |   |   |   |   |   |   |   |   |   |   |--- feature_0 <= 0.08
# |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
# |   |   |   |   |   |   |   |   |   |   |--- feature_0 >  0.08
# |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
# |   |   |   |   |   |   |--- feature_2 >  0.04
# |   |   |   |   |   |   |   |--- feature_5 <= 0.02
# |   |   |   |   |   |   |   |   |--- feature_1 <= 0.54
# |   |   |   |   |   |   |   |   |   |--- feature_3 <= -0.29
# |   |   |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |   |   |   |--- feature_3 >  -0.29
# |   |   |   |   |   |   |   |   |   |   |--- feature_5 <= 0.01
# |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |   |   |   |   |   |--- feature_5 >  0.01
# |   |   |   |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |   |   |--- feature_1 >  0.54
# |   |   |   |   |   |   |   |   |   |--- feature_0 <= 0.08
# |   |   |   |   |   |   |   |   |   |   |--- feature_3 <= -0.11
# |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
# |   |   |   |   |   |   |   |   |   |   |--- feature_3 >  -0.11
# |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
# |   |   |   |   |   |   |   |   |   |--- feature_0 >  0.08
# |   |   |   |   |   |   |   |   |   |   |--- feature_3 <= -0.25
# |   |   |   |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |   |   |   |   |--- feature_3 >  -0.25
# |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
# |   |   |   |   |   |   |   |--- feature_5 >  0.02
# |   |   |   |   |   |   |   |   |--- feature_5 <= 0.04
# |   |   |   |   |   |   |   |   |   |--- feature_2 <= 0.04
# |   |   |   |   |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |   |   |   |   |--- feature_2 >  0.04
# |   |   |   |   |   |   |   |   |   |   |--- feature_3 <= -0.27
# |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
# |   |   |   |   |   |   |   |   |   |   |--- feature_3 >  -0.27
# |   |   |   |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |   |   |--- feature_5 >  0.04
# |   |   |   |   |   |   |   |   |   |--- feature_5 <= 0.04
# |   |   |   |   |   |   |   |   |   |   |--- class: 3
# |   |   |   |   |   |   |   |   |   |--- feature_5 >  0.04
# |   |   |   |   |   |   |   |   |   |   |--- feature_1 <= 0.40
# |   |   |   |   |   |   |   |   |   |   |   |--- class: 3
# |   |   |   |   |   |   |   |   |   |   |--- feature_1 >  0.40
# |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
# |   |   |--- feature_5 >  0.07
# |   |   |   |--- feature_3 <= -0.31
# |   |   |   |   |--- class: 2
# |   |   |   |--- feature_3 >  -0.31
# |   |   |   |   |--- feature_1 <= 0.07
# |   |   |   |   |   |--- feature_4 <= -0.01
# |   |   |   |   |   |   |--- class: 0
# |   |   |   |   |   |--- feature_4 >  -0.01
# |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |--- feature_1 >  0.07
# |   |   |   |   |   |--- feature_3 <= -0.31
# |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |--- feature_3 >  -0.31
# |   |   |   |   |   |   |--- feature_2 <= 0.17
# |   |   |   |   |   |   |   |--- feature_5 <= 0.09
# |   |   |   |   |   |   |   |   |--- feature_4 <= 0.07
# |   |   |   |   |   |   |   |   |   |--- class: 0
# |   |   |   |   |   |   |   |   |--- feature_4 >  0.07
# |   |   |   |   |   |   |   |   |   |--- feature_3 <= -0.20
# |   |   |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |   |   |   |--- feature_3 >  -0.20
# |   |   |   |   |   |   |   |   |   |   |--- class: 3
# |   |   |   |   |   |   |   |--- feature_5 >  0.09
# |   |   |   |   |   |   |   |   |--- class: 3
# |   |   |   |   |   |   |--- feature_2 >  0.17
# |   |   |   |   |   |   |   |--- feature_3 <= -0.05
# |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |   |   |   |--- feature_3 >  -0.05
# |   |   |   |   |   |   |   |   |--- class: 3
# |   |--- feature_3 >  -0.02
# |   |   |--- feature_4 <= -0.00
# |   |   |   |--- feature_0 <= 0.06
# |   |   |   |   |--- feature_0 <= 0.01
# |   |   |   |   |   |--- feature_5 <= -0.03
# |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |--- feature_5 >  -0.03
# |   |   |   |   |   |   |--- class: 2
# |   |   |   |   |--- feature_0 >  0.01
# |   |   |   |   |   |--- class: 1
# |   |   |   |--- feature_0 >  0.06
# |   |   |   |   |--- feature_3 <= 0.01
# |   |   |   |   |   |--- class: 2
# |   |   |   |   |--- feature_3 >  0.01
# |   |   |   |   |   |--- class: 0
# |   |   |--- feature_4 >  -0.00
# |   |   |   |--- feature_3 <= 0.12
# |   |   |   |   |--- feature_2 <= -0.06
# |   |   |   |   |   |--- feature_5 <= -0.06
# |   |   |   |   |   |   |--- class: 0
# |   |   |   |   |   |--- feature_5 >  -0.06
# |   |   |   |   |   |   |--- class: 3
# |   |   |   |   |--- feature_2 >  -0.06
# |   |   |   |   |   |--- feature_2 <= 0.20
# |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |--- feature_2 >  0.20
# |   |   |   |   |   |   |--- feature_5 <= 0.04
# |   |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |   |--- feature_5 >  0.04
# |   |   |   |   |   |   |   |--- feature_4 <= 0.02
# |   |   |   |   |   |   |   |   |--- class: 0
# |   |   |   |   |   |   |   |--- feature_4 >  0.02
# |   |   |   |   |   |   |   |   |--- class: 2
# |   |   |   |--- feature_3 >  0.12
# |   |   |   |   |--- feature_2 <= 0.20
# |   |   |   |   |   |--- class: 2
# |   |   |   |   |--- feature_2 >  0.20
# |   |   |   |   |   |--- feature_3 <= 0.19
# |   |   |   |   |   |   |--- feature_1 <= 1.48
# |   |   |   |   |   |   |   |--- class: 1
# |   |   |   |   |   |   |--- feature_1 >  1.48
# |   |   |   |   |   |   |   |--- class: 3
# |   |   |   |   |   |--- feature_3 >  0.19
# |   |   |   |   |   |   |--- class: 3
