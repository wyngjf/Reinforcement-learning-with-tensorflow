"""
Base Class of rl algos.
"""

import numpy as np
import tensorflow as tf
import maths as maths
from gym.spaces import Box, Discrete


class RL:
    def __init__(
            self,
            env,
            lr_pi=0.01,
            # lr_v=0.01,
            gamma=0.99,
            lam=0.97,
            output_graph=False,
            seed=1,
            # ep_max=100,
            # ep_steps_max=4000,
            hidden_sizes=(64, 64),
            # train_v_iters=80,
            # train_pi_iters=80,
            # clip=0.2,
            # target_kl=0.01
    ):
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.lr_pi = lr_pi
        # self.lr_v = lr_v
        self.gamma = gamma
        # self.ep_max = ep_max
        # self.ep_steps_max = ep_steps_max
        # self.train_v_iters = train_v_iters
        # self.train_pi_iters = train_pi_iters
        self.lam = lam
        # self.clip = clip
        # self.target_kl = target_kl

        self.s = self._get_placeholder(env.observation_space, name='observations')
        self.a = self._get_placeholder(env.action_space, name='actions')
        print("observations: ", self.s)
        print("actions: ", self.a)
        self.placeholders, self.inputs_list = [], []
        self.placeholders.extend((self.s, self.a))
        self._build_net(hidden_sizes=hidden_sizes, action_space=env.action_space)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _get_placeholder(self, space, name):
        if isinstance(space, Box):
            shape = space.shape  # (act_dim, )
            dim = (None, ) if shape[0] == 1 else (None, *shape)
            # dim = (None, shape) if np.isscalar(shape) else (None, *shape)
            print("shape: ", shape, " dim: ", dim)
            return tf.placeholder(dtype=tf.float32, shape=dim, name=name)
        elif isinstance(space, Discrete):
            return tf.placeholder(dtype=tf.int32, shape=(None,), name=name)
        else:
            raise NotImplementedError

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
        act_logits = maths.mlp(s, list(hidden_sizes)+[act_dim], activation, output_activation)  # [[xx, ..., xx]]
        logps = tf.nn.log_softmax(act_logits)  # log prob. distribution of all the actions = log(soft_max) but faster
        pi = tf.squeeze(tf.multinomial(act_logits, 1), axis=1)  # sample one action from act_logits [x]
        # batch: list of probabilities [P_a0, P_a1, ...]
        logp_batch = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logps, axis=1)
        logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logps, axis=1)

        self.logps_old = tf.placeholder(dtype=tf.float32, shape=(None,act_dim), name='logps_old')
        kl_divergence = maths.discrete_kl_divergence(logps, self.logps_old)
        self.placeholders.append(self.logps_old)
        return pi, logp_pi, logp_batch, kl_divergence, [logps, ]

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

        l1 = tf.layers.dense(s, 5, tf.nn.relu, trainable=True)
        mu = tf.layers.dense(l1, act_dim, tf.nn.tanh, trainable=True)
        std = tf.layers.dense(l1, act_dim, tf.nn.softplus, trainable=True)
        log_std = tf.log(std)
        norm_dist = tf.distributions.Normal(loc=mu, scale=std)
        pi = tf.squeeze(norm_dist.sample(1), axis=0)

        #
        # mu = maths.mlp(s, list(hidden_sizes)+[act_dim], activation, output_activation)
        # log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(act_dim, dtype=np.float32))
        # std = tf.exp(log_std)
        # pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp_batch = maths.gaussian_likelihood(a, mu, log_std)
        logp_pi = maths.gaussian_likelihood(pi, mu, log_std)

        dim = (None, ) if act_dim == 1 else (None, act_dim)
        self.mu_old = tf.placeholder(dtype=tf.float32, shape=dim, name='mu_old')
        self.log_std_old = tf.placeholder(dtype=tf.float32, shape=dim, name='log_std_old')
        kl_divergence = maths.gaussian_kl_divergence(mu, log_std, self.mu_old, self.log_std_old)
        self.placeholders.extend((self.mu_old, self.log_std_old))
        return pi, logp_pi, logp_batch, kl_divergence, [mu, log_std]

    def _build_net(self, hidden_sizes=(30,30), activation=tf.tanh, output_activation=None, policy=None, action_space=None):
        pass

    def _choose_action(self, s):
        a = self.sess.run(self.pi, feed_dict={self.s: s[np.newaxis, :]})
        return a

    def _store_transition(self, s, a, r, v, p, ps):
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)
        self.ep_v.append(v)
        self.ep_logp_pi.append(p)
        self.ep_policy_status.append(ps)

    def _process_rollout(self, last_value):
        ep_r = np.append(self.ep_r, last_value)
        ep_v = np.append(self.ep_v, last_value)

        # calculate discounted returns
        ep_ret = maths.discounted_sum_vec(ep_r, self.gamma)[:-1]

        # calculate GAE and its normalization
        deltas = ep_r[:-1] + self.gamma * ep_v[1:] - ep_v[:-1]
        ep_adv = maths.discounted_sum_vec(deltas, self.gamma * self.lam)
        ep_adv -= np.mean(ep_adv)
        ep_adv /= np.std(ep_adv)
        return ep_ret, ep_adv

    # def _set_and_eval(self, x, alpha, old_params, inputs, step):
    #     self.sess.run(self.set_pi_params, feed_dict={self.v_ph: old_params - alpha * x * step})
    #     return self.sess.run([self.d_kl, self.pi_loss], feed_dict=inputs)
    #     # {self.s: np.vstack(self.ep_s),
    #     # self.a: np.squeeze(np.array(self.ep_a))})

    def _get_inputs(self, s_, r, done):
        # episodic states and actions
        ep_s = np.vstack(self.ep_s)
        ep_a = np.squeeze(np.array(self.ep_a))
        # episodic discounted returns and normalized GAEs
        v_ = r if done else self.sess.run(self.v, feed_dict={self.s: s_.reshape(1, -1)})
        ep_ret, ep_adv = self._process_rollout(v_)
        # episodic log probabilities of experienced actions
        ep_p = np.squeeze(np.array(self.ep_logp_pi))
        # episodic policy status, depend on type of the chosen policy
        ep_ps = np.vstack(self.ep_policy_status)
        ep_policy_status = [np.squeeze(ep_ps[:, i]) for i in range(ep_ps.shape[1])]

        return [ep_s, ep_a, ep_ret, ep_adv, ep_p] + ep_policy_status

    def _update(self, inputs):
        pass

    def train(self, env, render_threshold_reward, render=False):
        for ep_index in range(self.ep_max):
            s = env.reset()
            for step_index in range(self.ep_steps_max):
                if render:
                    env.render()
                agent_status = self.sess.run(self.agent_status, feed_dict={self.s: s[np.newaxis, :]})
                # print('epoch: %d, time: %d: ' % (ep_index, step_index), agent_status)
                a, logp_pi, v, policy_status = agent_status[0], agent_status[1], agent_status[2], agent_status[3:]
                # print('epoch: %d, time: %d: ' % (ep_index, step_index), policy_status, s)
                s_, r, done, _ = env.step(a[0])

                self._store_transition(s, a, r, v, logp_pi, policy_status)

                terminal = done or (step_index == self.ep_steps_max-1)
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

                    inputs = self._get_inputs(s_, r, done)
                    print("episode:", ep_index, inputs[4])
                    self._update(inputs)
                    done = False

                    break

                s = s_
