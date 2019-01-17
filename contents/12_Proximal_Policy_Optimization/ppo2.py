"""
PPO based on RL base class
"""

import numpy as np
import tensorflow as tf
import maths as maths
from rl_core import RL
from gym.spaces import Box, Discrete


class PPO(RL):
    def __init__(
            self,
            env,
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
    ):
        self.lr_v = lr_v
        self.ep_max = ep_max
        self.ep_steps_max = ep_steps_max
        self.train_v_iters = train_v_iters
        self.train_pi_iters = train_pi_iters
        self.clip = clip
        self.target_kl = target_kl
        super(PPO, self).__init__(env, lr_pi, gamma, lam, output_graph, seed, hidden_sizes)

    def _build_net(self, hidden_sizes=(30, 30), activation=tf.tanh, output_activation=None,
                   policy=None, action_space=None):
        # to be stored during the episode
        print("building net")
        self.ep_s, self.ep_a, self.ep_r, self.ep_v = [], [], [], []
        self.ep_logps, self.ep_logp_pi = [], []
        self.ep_policy_status = []

        # placeholders for calculation of loss functions
        self.ep_ret = tf.placeholder(dtype=tf.float32, shape=(None,), name='ep_returns')
        self.ep_adv = tf.placeholder(dtype=tf.float32, shape=(None,), name='ep_advantages')
        self.logp_old = tf.placeholder(dtype=tf.float32, shape=(None,), name='logp_old')
        self.placeholders.extend((self.ep_ret, self.ep_adv, self.logp_old))

        # construct actor-critic networks
        # default policy
        if policy is None and isinstance(action_space, Box):
            policy = self._mlp_gaussian_policy
        elif policy is None and isinstance(action_space, Discrete):
            policy = self._mlp_discrete_policy

        with tf.variable_scope('actor'):  # TODO
            self.pi, self.logp_pi, logp_batch, self.d_kl, self.pi_status = \
                policy(self.s, self.a, hidden_sizes, activation, output_activation, action_space)
        with tf.variable_scope('critic'):
            # self.v = tf.squeeze(self._mlp(self.s, list(hidden_sizes)+[1], activation, None), axis=1)
            self.v = tf.squeeze(maths.mlp(self.s, (30, 1), activation, None), axis=1)
        self.agent_status = [self.pi, self.logp_pi, self.v] + self.pi_status

        self.test = logp_batch

        # construct loss functions for actor and critic
        ratio = tf.exp(logp_batch - self.logp_old)
        min_adv = tf.where(self.ep_adv>0, (1+self.clip)*self.ep_adv, (1-self.clip)*self.ep_adv)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.ep_adv, min_adv))
        self.v_loss = tf.reduce_mean((self.ep_ret - self.v)**2)
        with tf.name_scope('train'):
            self.train_pi = tf.train.AdamOptimizer(self.lr_pi).minimize(self.pi_loss)
            self.train_v = tf.train.AdamOptimizer(self.lr_v).minimize(self.v_loss)

        # construct KL divergence
        self.d_kl = tf.reduce_mean(logp_batch - self.logp_old)

    def _update(self, inputs):
        self.inputs = {k: v for k, v in zip(self.placeholders, inputs)}
        for i in range(self.train_pi_iters):
            test, d_kl, _ = self.sess.run([self.test, self.d_kl, self.train_pi], feed_dict=self.inputs)
            # print(d_kl, 1.5*self.target_kl)
            if d_kl > 1.5 * self.target_kl:
                print('KL divergence %f exceed threshold %f, early stop at step %d!' % (d_kl, self.target_kl, i))
                break

        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed_dict=self.inputs)

        self.ep_s, self.ep_a, self.ep_r, self.ep_v = [], [], [], []
        self.ep_logps, self.ep_logp_pi = [], []    # empty episode data

