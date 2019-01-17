"""
Maths
"""

import numpy as np
import tensorflow as tf
import scipy.signal

EPS = 1e-8


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def discrete_kl_divergence(logp1, logp2):
    return tf.squeeze(tf.reduce_sum(tf.exp(logp1) * (logp1 - logp2), axis=1))


def gaussian_kl_divergence(mu1, log_std1, mu2, log_std2):
    var1, var2 = tf.exp(2 * log_std1), tf.exp(2 * log_std2)
    vec_sum = log_std2 - log_std1 + 0.5 * (((mu1 - mu2)**2 + var1)/(var2 + EPS) - 1)
    return tf.squeeze(tf.reduce_sum(vec_sum, axis=1))


def mlp(x, hidden_sizes=(64,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]


def flat_concat(xs):
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)


def flat_grad(f, params):
    grads = tf.gradients(xs=params, ys=f)
    return tf.concat([tf.reshape(x, (-1,)) for x in grads], axis=0)


def flat_size(x):
    return int(np.prod(x.shape.as_list()))  # get number of params in a tensor


def assign_params_from_flat(x, params):
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])


def hessian_vector_product(f, params):
    # for H = grad**2 f, compute Hx
    g = flat_grad(f, params)
    x = tf.placeholder(tf.float32, shape=g.shape)
    return x, flat_grad(tf.reduce_sum(g * x), params)


def discounted_sum_vec(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def conjugate_gradient(Ax, b):
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
