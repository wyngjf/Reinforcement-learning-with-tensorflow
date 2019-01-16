"""
vanila policy gradient
"""

import numpy as np
import scipy.signal
import tensorflow as tf
from gym.spaces import Box, Discrete

class VPG_GAE:
    def __init__(
            self,
            env,
            lr_pi=0.01,
            lr_v=0.01,
            gamma=0.99,
            lam=0.97,
            output_graph=False,
            seed=1,
            ep_max=1000,
            ep_steps_max=1000,
            hidden_sizes=(64, 64),
            train_v_iters=80
    ):
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.lr_pi = lr_pi
        self.lr_v = lr_v
        self.gamma = gamma
        self.ep_max=ep_max
        self.ep_steps_max=ep_steps_max
        self.train_v_iters=train_v_iters
        self.lam=lam

        self.s = self._get_placeholder(env.observation_space, name='observations')
        print("observations: ", self.s)
        self.a = self._get_placeholder(env.action_space, name='actions')
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
            dim = (None, shape) if np.isscalar(shape) else (None, *shape)
            return tf.placeholder(dtype=tf.float32, shape=dim, name=name)
        elif isinstance(space, Discrete):
            return tf.placeholder(dtype=tf.int32, shape=(None,), name=name)
        else:
            raise NotImplementedError

    def _gaussian_likelihood(self, x, mu, log_std):
        eps = 1e-8
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + eps)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    def _mlp(self, x, hidden_sizes=(64,), activation=tf.tanh, output_activation=None):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, units=h, activation=activation)
        return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

    def _mlp_discrete_policy(self, s, a, hidden_sizes, activation, output_activation, action_space):
        """
        generate a policy network for the discrete case
        :param s: state placeholder
        :param a: action placeholder, e.g. input the action series as list [a_1, ..., a_T]
        :param hidden_sizes: list [l1, l2, ...]
        :param activation:
        :param output_activation:
        :param action_space: env.action_space
        :return:
            pi: the action chosen by the current policy at state s
            logp_batch: the list of log probability corresponding to the list of actions a
            logp_pi: the log probability that pi is chosen
        """
        act_dim = action_space.n
        act_logits = self._mlp(s, list(hidden_sizes)+[act_dim], activation, None)  # [[xx, ..., xx]]
        logps = tf.nn.log_softmax(act_logits)  # log prob. distribution of all the actions = log(soft_max) but faster
        pi = tf.squeeze(tf.multinomial(act_logits, 1), axis=1)  # sample one action from act_logits [x]
        # batch: list of probabilities [P_a0, P_a1, ...]
        logp_batch = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logps, axis=1)
        logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logps, axis=1)
        return pi, logp_batch, logp_pi

    def _mlp_gaussian_policy(self, s, a, hidden_sizes, activation, output_activation, action_space):
        """
        generate a policy network for the continuous case
        :param s: state placeholder
        :param a: action placeholder, e.g. input the action matrix
            a = [[a_11, ..., a_D1],
                 [a_12, ..., a_D2],
                 ...
                 [a_1T, ..., a_DT]]
        :param hidden_sizes: list [l1, l2, ...]
        :param activation:
        :param output_activation:
        :param action_space: env.action_space
        :return:
            pi: the action chosen by the current policy at state s
            logp_batch: the list of log probability corresponding to the list of actions a
            logp_pi: the log probability that pi is chosen
        """
        act_dim = action_space.shape[0]
        mu = self._mlp(s, list(hidden_sizes)+[act_dim], activation, output_activation)
        log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(act_dim, dtype=np.float32))
        std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp_batch = self._gaussian_likelihood(a, mu, log_std)
        logp_pi = self._gaussian_likelihood(pi, mu, log_std)
        return pi, logp_batch, logp_pi

    def _build_net(self, hidden_sizes=(30,30), activation=tf.tanh, output_activation=None, policy=None, action_space=None):
        self.ep_s, self.ep_a, self.ep_r, self.ep_v = [], [], [], []
        self.ep_ret = tf.placeholder(dtype=tf.float32, shape=(None,), name='ep_returns')
        self.ep_adv = tf.placeholder(dtype=tf.float32, shape=(None,), name='ep_advantages')
        print("ep_returns: ", self.ep_ret)

        # default policy
        if policy is None and isinstance(action_space, Box):
            policy = self._mlp_gaussian_policy
        elif policy is None and isinstance(action_space, Discrete):
            policy = self._mlp_discrete_policy

        with tf.variable_scope('actor'):
            self.pi, logp_batch, _ = policy(self.s, self.a, hidden_sizes, activation, output_activation, action_space)
        with tf.variable_scope('critic'):
            # self.v = tf.squeeze(self._mlp(self.s, list(hidden_sizes)+[1], activation, None), axis=1)
            self.v = tf.squeeze(self._mlp(self.s, (30, 1), activation, None), axis=1)

        pi_loss = -tf.reduce_mean(logp_batch * self.ep_adv)
        v_loss = tf.reduce_mean((self.ep_ret - self.v)**2)
        with tf.name_scope('train'):
            self.train_pi = tf.train.AdamOptimizer(self.lr_pi).minimize(pi_loss)
            self.train_v = tf.train.AdamOptimizer(self.lr_v).minimize(v_loss)

    def _choose_action(self, s):
        a = self.sess.run(self.pi, feed_dict={self.s: s[np.newaxis, :]})
        return a

    def _store_transition(self, s, a, r, v):
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)
        self.ep_v.append(v)

    def _process_rollout(self, last_value):
        ep_r = np.append(self.ep_r, last_value)
        ep_v = np.append(self.ep_v, last_value)

        ep_ret = self._discounted_sum_vec(ep_r, self.gamma)[:-1]
        ep_ret -= np.mean(ep_ret)
        ep_ret /= np.std(ep_ret)

        deltas = ep_r[:-1] + self.gamma * ep_v[1:] - ep_v[:-1]
        ep_adv = self._discounted_sum_vec(deltas, self.gamma * self.lam)
        return ep_ret, ep_adv

    def _discounted_sum_vec(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def _update(self, s_):
        v_ = self.sess.run(self.v, feed_dict={self.s: s_.reshape(1, -1)})
        # discount and normalize episode reward
        discounted_ep_return, ep_adv = self._process_rollout(v_)
        # print('discounted_ep_return: ', discounted_ep_return)

        # train on episode
        self.sess.run(self.train_pi, feed_dict={
             self.s: np.vstack(self.ep_s),              # shape=(None, s_dim)
             self.a: np.squeeze(np.array(self.ep_a)),   # shape=(None, a_dim)
             self.ep_adv: ep_adv,         # shape=(None, )
        })

        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed_dict={
                self.s: np.vstack(self.ep_s),
                self.ep_ret: discounted_ep_return
            })

        self.ep_s, self.ep_a, self.ep_r, self.ep_v = [], [], [], []    # empty episode data

    def train(self, env, render_threshold_reward, render=False):
        for ep_index in range(self.ep_max):
            s = env.reset()
            for step_index in range(self.ep_steps_max):
                if render:
                    env.render()
                a = self._choose_action(s)

                s_, r, done, _ = env.step(a[0])
                v = self.sess.run(self.v, feed_dict={self.s: s.reshape(1, -1)})

                self._store_transition(s, a, r, v)

                terminal = done or ((step_index+1) == self.ep_steps_max)
                if terminal:
                    # calculate running reward
                    ep_rs_sum = sum(self.ep_r)
                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                    if running_reward > render_threshold_reward:
                        render = True     # rendering
                    print("episode:", ep_index, "  reward:", int(running_reward))

                    self._update(s_)
                    break

                s = s_
