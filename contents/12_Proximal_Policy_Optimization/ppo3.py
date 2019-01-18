"""
TRPO
"""

import numpy as np
import scipy.signal
import tensorflow as tf
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt

EPS = 1e-8

class PPO:
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
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.lr_pi = lr_pi
        self.lr_v = lr_v
        self.gamma = gamma
        self.ep_max = ep_max
        self.ep_steps_max = ep_steps_max
        self.train_v_iters = train_v_iters
        self.train_pi_iters = train_pi_iters
        self.lam = lam
        self.clip = clip
        self.target_kl = target_kl

        self.s = self._get_placeholder(env.observation_space, name='observations')
        self.a = self._get_placeholder(env.action_space, name='actions')
        print("observations: ", self.s)
        print("actions: ", self.a)

        self._build_net(hidden_sizes=hidden_sizes, action_space=env.action_space)

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _get_placeholder(self, space, name):
        if isinstance(space, Box):
            shape = space.shape  # (act_dim, )
            dim = (None,) if shape[0] == 1 else (None, *shape)
            # dim = (None, shape) if np.isscalar(shape) else (None, *shape)
            return tf.placeholder(dtype=tf.float32, shape=dim, name=name)
        elif isinstance(space, Discrete):
            return tf.placeholder(dtype=tf.int32, shape=(None,), name=name)
        else:
            raise NotImplementedError

    def _build_net(self, hidden_sizes=(30,30), activation=tf.tanh, output_activation=None, policy=None, action_space=None):
        # to be stored during the episode
        self.ep_s, self.ep_a, self.ep_r, self.ep_v, self.ep_logp_pi = [], [], [], [], []

        # placeholders for calculation of loss functions
        self.ep_ret = tf.placeholder(dtype=tf.float32, shape=(None,), name='ep_returns')
        self.ep_adv = tf.placeholder(dtype=tf.float32, shape=(None,), name='ep_advantages')
        self.logp_old = tf.placeholder(dtype=tf.float32, shape=(None,), name='logp_old')

        act_dim = action_space.shape[0]
        with tf.variable_scope('actor'):  # TODO
            l1 = tf.layers.dense(self.s, 100, tf.nn.relu, trainable=True)
            mu = 2 * tf.layers.dense(l1, act_dim, tf.nn.tanh, trainable=True)
            std = tf.layers.dense(l1, act_dim, tf.nn.softplus, trainable=True)
            norm_dist = tf.distributions.Normal(loc=mu, scale=std)
            # self.pi = tf.squeeze(norm_dist.sample(1), axis=0)
            self.pi = mu + tf.random_normal(tf.shape(mu)) * std
            self.logp_pi = norm_dist.prob(self.pi)
            logp_batch = norm_dist.prob(self.a)

        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.s, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)

        ratio = tf.exp(logp_batch - self.logp_old)
        min_adv = tf.clip_by_value(ratio, 1.-self.clip, 1.+self.clip)*self.ep_adv
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.ep_adv, min_adv))
        self.v_loss = tf.reduce_mean((self.ep_ret - self.v)**2)
        with tf.name_scope('train'):
            self.train_pi = tf.train.AdamOptimizer(self.lr_pi).minimize(self.pi_loss)
            self.train_v = tf.train.AdamOptimizer(self.lr_v).minimize(self.v_loss)

        self.d_kl = tf.reduce_mean(self.logp_old - logp_batch)

    def _choose_action(self, s):
        a = self.sess.run(self.pi, feed_dict={self.s: s[np.newaxis, :]})
        return a

    def _get_agent_status(self, s):
        a, logp_pi, v = self.sess.run([self.pi, self.logp_pi, self.v],
                                      feed_dict={self.s: s[np.newaxis, :]})
        return a, logp_pi, v,

    def _store_transition(self, s, a, r, v, p):
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)
        self.ep_v.append(v)
        self.ep_logp_pi.append(p)

    def _process_rollout(self, s_, r, done):
        v_ = r if done else self.sess.run(self.v, feed_dict={self.s: s_.reshape(1, -1)})
        ep_r = np.append(self.ep_r, v_)
        ep_v = np.append(self.ep_v, v_)

        ep_ret = self._discounted_sum_vec(ep_r, self.gamma)[:-1]

        deltas = ep_r[:-1] + self.gamma * ep_v[1:] - ep_v[:-1]
        ep_adv = self._discounted_sum_vec(deltas, self.gamma * self.lam)
        ep_adv -= np.mean(ep_adv)
        ep_adv /= np.std(ep_adv)
        return ep_ret, ep_adv

    def _discounted_sum_vec(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def _update(self):
        ep_s = np.vstack(self.ep_s)
        ep_a = np.squeeze(np.array(self.ep_a))
        ep_p = np.squeeze(np.array(self.ep_logp_pi))
        self.inputs = {self.s: ep_s,
                       self.a: ep_a,
                       self.ep_adv: self.ep_Adv,
                       self.ep_ret: self.ep_G,
                       self.logp_old: ep_p
                       }
        for i in range(self.train_pi_iters):
            d_kl, _ = self.sess.run([self.d_kl, self.train_pi], feed_dict=self.inputs)
            if d_kl > 1.5 * self.target_kl:
                print('KL divergence %f exceed threshold %f, early stop at step %d!' % (d_kl, self.target_kl, i))
                break

        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed_dict=self.inputs)

        self.ep_s, self.ep_a, self.ep_r, self.ep_v = [], [], [], []
        self.ep_logp_pi = []    # empty episode data

    def train(self, env, render_threshold_reward, render=False):
        self.all_rew = []
        for ep_index in range(self.ep_max):
            ep_r = 0
            s = env.reset()
            for step_index in range(self.ep_steps_max):
                if render:
                    env.render()
                a, logp_pi, v = self._get_agent_status(s)
                a = np.clip(a, -2, 2)
                s_, r, done, _ = env.step(a[0])
                ep_r += r

                self._store_transition(s, a, (r+8.)/8., v, logp_pi)

                terminal = done or (step_index == self.ep_steps_max-1)
                if terminal:
                    # # calculate running reward
                    # ep_rs_sum = sum(self.ep_r)
                    # if 'running_reward' not in globals():
                    #     running_reward = ep_rs_sum
                    # else:
                    #     running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                    # if running_reward > render_threshold_reward:
                    #     render = True     # rendering
                    # print("episode:", ep_index, "  reward:", int(running_reward))

                    self.ep_G, self.ep_Adv = self._process_rollout(s_, r, done)
                    self._update()
                    done = False
                    break

                s = s_
            if ep_index == 0:
                self.all_rew.append(ep_r)
            else:
                self.all_rew.append(self.all_rew[-1] * 0.9 + ep_r * 0.1)
            # if ep_r > -100:
            if ep_index > 995:
                render = True
            print(
                'Ep: %i' % ep_index,
                "|Ep_r: %f" % ep_r
            )
        plt.plot(np.arange(len(self.all_rew)), self.all_rew)
        plt.xlabel('Episode')
        plt.ylabel('Moving averaged episode reward')
        plt.show()
