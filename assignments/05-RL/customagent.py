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


# class Replay:
#     def __init__(self, max_size):
#         self.max_size = max_size
#         self.counter = 0
#         self.s = np.zeros((self.max_size, 8), dtype=np.float32)
#         self.a = np.zeros(self.max_size, dtype=np.int32)
#         self.r = np.zeros(self.max_size, dtype=np.float32)
#         self.next_s = np.zeros((self.max_size, 8), dtype=np.float32)
#         self.terminated = np.zeros(self.max_size, dtype=np.bool_)

#     def update1(self, s, a):
#         idx = self.counter % self.size
#         self.s[idx] = s
#         self.a[idx] = a


#     def update2(self, r, next_s, terminated):
#         idx = self.counter % self.size
#         self.r[idx] = r
#         self.next_s[idx] = next_s
#         self.terminated[idx] = terminated
#         self.counter += 1

#     def sample_batch(self, batch_size):
#         max_buffer = min(self.counter, self.max_size)
#         batch = np.random.choice(max_buffer, batch_size, replace=False)
#         state_batch = self.s[batch]
#         action_batch = self.a[batch]
#         reward_batch = self.r[batch]
#         new_state_batch = self.next_s[batch]
#         done_batch = self.terminated[batch]

#         return state_batch, action_batch, reward_batch, new_state_batch, done_batch


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
        # self.target_net=Net(8,4)
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)

        # self.replay_buffer=Replay(5000)

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

        # self.replay_buffer.update1(observation,action)
        self.state = observation
        return action

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

        # store new sample to replay buffer
        # self.replay_buffer.update2(reward,observation,terminated or truncated)
        # sample from replay buffer
        # if self.replay_buffer.counter<self.batch_size:
        #     return
        # s_batch,a_batch,r_batch,next_s_batch,terminated_batch=self.replay_buffer.sample_batch(self.batch_size)

        q_pred = self.net(self.state).max(dim=1)
        q_next_max = torch.max(self.net(observation), 1)
        q_target = reward + self.discount * q_next_max * (1 - terminated)

        L = nn.CrossEntropyLoss()
        loss = L(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return
