"""
Maths
"""

import numpy as np
import tensorflow as tf

EPS = 1e-8


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def discrete_kl_divergence(logp1, logp2):
    return tf.squeeze(tf.reduce_sum(tf.exp(logp1) * (logp1 - logp2), axis=1))