"""
Policy Gradient, Reinforcement Learning.

The MountainCar Example
"""

import gym
from ppo2 import PPO

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

agent = PPO(
    env=env,
    lr_pi=0.01,
    lr_v=0.01,
    gamma=0.99,
    lam=0.97,
    output_graph=False,
    seed=1,
    ep_max=2000,
    ep_steps_max=1000,
    hidden_sizes=(30,),
    train_v_iters=80,
    train_pi_iters=80,
    clip=0.2,
    target_kl=0.01
)

# CartPole hidden_size=(7,), v_iter = pi_iter = 80, clip=0.2, kl = 0.01
# RL.train(env, render_threshold_reward=500, render=False)

# MountainCar hidden_size=(7,), v_iter = pi_iter = 80, clip=0.2, kl = 0.01
# RL.train(env, render_threshold_reward=-200, render=False)

# Pendulum hidden_size=(30,), v_iter = pi_iter = 20, clip=0.2, kl = 0.01
agent.train(env, render_threshold_reward=-2000, render=False)
