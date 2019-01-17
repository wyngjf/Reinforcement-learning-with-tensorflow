"""
Policy Gradient, Reinforcement Learning.

The MountainCar Example
"""

import gym
from vpg import VPG
from vpg_ac_gae import VPG_GAE
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is larger than this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

RENDER = False  # rendering wastes time

# env = gym.make('MountainCar-v0')
# env = gym.make('CartPole-v0')
env = gym.make('Pendulum-v0')
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

RL = VPG_GAE(
    env=env,
    lr_pi=0.01,
    lr_v=0.01,
    gamma=0.995,
    lam=0.97,
    output_graph=False,
    seed=1,
    ep_max=3000,
    ep_steps_max=1000,
    hidden_sizes=(30,),
    train_v_iters=80
)

RL.train(env, render_threshold_reward=-100, render=False)
# RL.train(env, render_threshold_reward=1000, render=False)
