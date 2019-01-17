"""
TRPO
"""

import numpy as np
import scipy.signal
import tensorflow as tf
from gym.spaces import Box, Discrete
from scipy.sparse.linalg import cg

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

    def _gaussian_likelihood(self, x, mu, log_std):
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    def _discrete_kl_divergence(self, logp1, logp2):
        return tf.squeeze(tf.reduce_sum(tf.exp(logp1) * (logp1 - logp2), axis=1))

    def _gaussian_kl_divergence(self, mu1, log_std1, mu2, log_std2):
        var1, var2 = tf.exp(2 * log_std1), tf.exp(2 * log_std2)
        vec_sum = log_std2 - log_std1 + 0.5 * (((mu1 - mu2)**2 + var1)/(var2 + EPS) - 1)
        return tf.squeeze(tf.reduce_sum(vec_sum, axis=1))

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

        self.logps_old = tf.placeholder(dtype=tf.float32, shape=(None,act_dim), name='logps_old')
        kl_divergence = self._discrete_kl_divergence(logps, self.logps_old)
        return pi, logp_batch, logp_pi  #, logps, kl_divergence

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

        l1 = tf.layers.dense(s, 100, tf.nn.relu, trainable=True)
        mu = tf.layers.dense(l1, act_dim, tf.nn.tanh, trainable=True)
        std = tf.layers.dense(l1, act_dim, tf.nn.softplus, trainable=True)
        norm_dist = tf.distributions.Normal(loc=mu, scale=std)
        pi = tf.squeeze(norm_dist.sample(1), axis=0)
        logp_pi = norm_dist.prob(pi)
        logp_batch = norm_dist.prob(a)


        # mu = self._mlp(s, list(hidden_sizes)+[act_dim], activation, output_activation)
        # log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(act_dim, dtype=np.float32))
        # std = tf.exp(log_std)
        # pi = mu + tf.random_normal(tf.shape(mu)) * std
        # logp_batch = self._gaussian_likelihood(a, mu, log_std)
        # logp_pi = self._gaussian_likelihood(pi, mu, log_std)

        # mu_old = tf.placeholder(dtype=tf.float32, shape=(None,act_dim), name='mu_old')
        # log_std_old = tf.placeholder(dtype=tf.float32, shape=(None,act_dim), name='log_std_old')
        # kl_divergence = self._gaussian_kl_divergence(mu, log_std, mu_old, log_std_old)
        return pi, logp_batch, logp_pi  #, mu, log_std, kl_divergence

    def _build_net(self, hidden_sizes=(30,30), activation=tf.tanh, output_activation=None, policy=None, action_space=None):
        # to be stored during the episode
        self.ep_s, self.ep_a, self.ep_r, self.ep_v = [], [], [], []
        self.ep_logps, self.ep_logp_pi = [], []

        # placeholders for calculation of loss functions
        self.ep_ret = tf.placeholder(dtype=tf.float32, shape=(None,), name='ep_returns')
        self.ep_adv = tf.placeholder(dtype=tf.float32, shape=(None,), name='ep_advantages')
        self.logp_old = tf.placeholder(dtype=tf.float32, shape=(None,), name='logp_old')

        # default policy
        if policy is None and isinstance(action_space, Box):
            policy = self._mlp_gaussian_policy
        elif policy is None and isinstance(action_space, Discrete):
            policy = self._mlp_discrete_policy

        with tf.variable_scope('actor'):  # TODO
            # self.pi, logp_batch, self.logp_pi, self.logps, self.d_kl = \
            self.pi, logp_batch, self.logp_pi = \
                policy(self.s, self.a, hidden_sizes, activation, output_activation, action_space)
        with tf.variable_scope('critic'):
            # self.v = tf.squeeze(self._mlp(self.s, list(hidden_sizes)+[1], activation, None), axis=1)
            self.v = tf.squeeze(self._mlp(self.s, list(hidden_sizes)+[1], activation, None), axis=1)

        ratio = tf.exp(logp_batch - self.logp_old)
        min_adv = tf.where(self.ep_adv>0, (1+self.clip)*self.ep_adv, (1-self.clip)*self.ep_adv)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.ep_adv, min_adv))
        self.v_loss = tf.reduce_mean((self.ep_ret - self.v)**2)
        with tf.name_scope('train'):
            self.train_pi = tf.train.AdamOptimizer(self.lr_pi).minimize(self.pi_loss)
            self.train_v = tf.train.AdamOptimizer(self.lr_v).minimize(self.v_loss)

        self.d_kl = tf.reduce_mean(self.logp_old - logp_batch)

    def _get_vars(self, scope=''):
        return [x for x in tf.trainable_variables() if scope in x.name]

    def _flat_concat(self, xs):
        return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)

    def _flat_grad(self, f, params):
        grads = tf.gradients(xs=params, ys=f)
        return tf.concat([tf.reshape(x, (-1,)) for x in grads], axis=0)

    def _hessian_vector_product(self, f, params):
        # for H = grad**2 f, compute Hx
        g = self._flat_grad(f, params)
        x = tf.placeholder(tf.float32, shape=g.shape)
        return x, self._flat_grad(tf.reduce_sum(g * x), params)

    def _assign_params_from_flat(self, x, params):
        flat_size = lambda p: int(np.prod(p.shape.as_list()))  # get number of params in a tensor
        splits = tf.split(x, [flat_size(p) for p in params])
        new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
        return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])

    def _choose_action(self, s):
        a = self.sess.run(self.pi, feed_dict={self.s: s[np.newaxis, :]})
        return a

    def _get_agent_status(self, s):
        # a, logp_pi, logps, v = self.sess.run([self.pi, self.logp_pi, self.logps, self.v],
        #                                      feed_dict={self.s: s[np.newaxis, :]})
        a, logp_pi, v = self.sess.run([self.pi, self.logp_pi, self.v],
                                      feed_dict={self.s: s[np.newaxis, :]})
        return a, logp_pi, v,

    def _store_transition(self, s, a, r, v, p, ps):
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)
        self.ep_v.append(v)
        self.ep_logp_pi.append(p)
        self.ep_logps.append(ps)

    def _process_rollout(self, last_value):
        ep_r = np.append(self.ep_r, last_value)
        ep_v = np.append(self.ep_v, last_value)

        ep_ret = self._discounted_sum_vec(ep_r, self.gamma)[:-1]
        # ep_ret -= np.mean(ep_ret)
        # ep_ret /= np.std(ep_ret)

        deltas = ep_r[:-1] + self.gamma * ep_v[1:] - ep_v[:-1]
        ep_adv = self._discounted_sum_vec(deltas, self.gamma * self.lam)
        ep_adv -= np.mean(ep_adv)
        ep_adv /= np.std(ep_adv)
        return ep_ret, ep_adv

    def _discounted_sum_vec(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def _conjugate_gradient(self, Ax, b):
        x = np.zeros_like(b)
        r = b.copy()  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.copy()
        r_dot_old = np.dot(r, r)
        for _ in range(self.cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def _set_and_eval(self, x, alpha, old_params, inputs, step):
        self.sess.run(self.set_pi_params, feed_dict={self.v_ph: old_params - alpha * x * step})
        return self.sess.run([self.d_kl, self.pi_loss], feed_dict=inputs)
        # {self.s: np.vstack(self.ep_s),
        # self.a: np.squeeze(np.array(self.ep_a))})

    def _update(self):
        ep_s = np.vstack(self.ep_s)
        ep_a = np.squeeze(np.array(self.ep_a))
        ep_p = np.squeeze(np.array(self.ep_logp_pi))
        # ep_logps = np.vstack(self.ep_logps)
        self.inputs = {self.s: ep_s,
                       self.a: ep_a,
                       self.ep_adv: self.ep_Adv,
                       self.ep_ret: self.ep_G,
                       # self.logps_old: ep_logps,
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
        self.ep_logps, self.ep_logp_pi = [], []    # empty episode data

    def train(self, env, render_threshold_reward, render=False):
        for ep_index in range(self.ep_max):
            s = env.reset()
            for step_index in range(self.ep_steps_max):
                if render:
                    env.render()
                a, logp_pi, v = self._get_agent_status(s)

                s_, r, done, _ = env.step(a[0])

                self._store_transition(s, a, r, v, logp_pi, [1])

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

                    v_ = r if done else self.sess.run(self.v, feed_dict={self.s: s_.reshape(1, -1)})
                    done = False
                    self.ep_G, self.ep_Adv = self._process_rollout(v_)
                    self._update()

                    break

                s = s_
