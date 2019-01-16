"""
Policy Gradient, Reinforcement Learning.

The MountainCar Example
"""

import gym
from ppo import PPO
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is larger than this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

RENDER = False  # rendering wastes time

# env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v0')
# env = gym.make('Pendulum-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

# Discrete(3) action_space.n = 3
# Box(2,) observation_space.shape = (2,)
# [0.6  0.07]
# [-1.2  -0.07]


# RL = VPG(
#     env=env,
#     learning_rate=0.02,
#     gamma=0.995,
#     output_graph=False,
#     seed=1,
#     ep_max=3000,
#     ep_steps_max=8000,
#     hidden_sizes=(30,)
# )

RL = PPO(
    env=env,
    lr_pi=0.01,
    lr_v=0.01,
    gamma=0.99,
    lam=0.97,
    output_graph=False,
    seed=1,
    ep_max=100,
    ep_steps_max=4000,
    hidden_sizes=(64, 64),
    train_v_iters=80,
    train_pi_iters=80,
    clip=0.2,
    target_kl=0.01
)

RL.train(env, render_threshold_reward=500, render=False)
# RL.train(env, render_threshold_reward=-200, render=False)
