import gymnasium as gym
from customagent import Agent

# import time

SHOW_ANIMATIONS = True

env = gym.make("LunarLander-v2", render_mode="human" if SHOW_ANIMATIONS else "none")
observation, info = env.reset(seed=42)

agent = Agent(
    action_space=env.action_space,
    observation_space=env.observation_space,
)

total_reward = 0
last_n_rewards = []
for _ in range(100000):
    action = agent.act(observation)
    observation, reward, terminated, _ = env.step(action)
    agent.learn(observation, reward, terminated)
    total_reward += reward

    if terminated:
        observation, info = env.reset()
        last_n_rewards.append(total_reward)
        n = min(30, len(last_n_rewards))
        avg = sum(last_n_rewards[-n:]) / n
        total_reward = 0

env.close()
