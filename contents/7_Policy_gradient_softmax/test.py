import numpy as np
import tensorflow as tf
act_dim =3
# A = np.array([2., 0., 0., 1.])
B = np.array([[2.], [0.], [0.], [1.]])
A = tf.convert_to_tensor(B, dtype=tf.int32)
A = tf.squeeze(A)
a = np.array([[0.09392077, 0.03547905, 0.00434585]])
sess = tf.Session()

logits = tf.convert_to_tensor(a, dtype=tf.float32)
logp_all = tf.nn.log_softmax(logits)
t = tf.multinomial(logits,1)
pi = tf.squeeze(t, axis=1)
d = tf.one_hot(pi, depth=act_dim)
e = d * logp_all
logp_pi = tf.reduce_sum(e, axis=1)

b = tf.one_hot(A, depth=act_dim)
c = b * logp_all
logp = tf.reduce_sum(c, axis=1)

log_all, t, pi, b, c, logp, d, e, logp_pi = sess.run([logp_all, t, pi, b, c, logp, d, e, logp_pi])
print('logp_all:', log_all)
print('t:', t)
print('pi:', pi)
print('b:', b)
print('c:', c)
print('logp:', logp)
print('d:', d)
print('e:', e)
print('logp_pi:', logp_pi)

# l = np.array([[0.1, 0.5, 0.4]])
# m = np.array([[0.2, 0.5, 0.3]])
# n = tf.exp(l) * (l - m)
# o = tf.reduce_sum(n, axis=1)
# p = tf.squeeze(o)
# nn, oo, pp = sess.run([n, o, p])
# print(nn, oo, pp)

# def flat_concat(xs):
#     return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)
# C = np.array([[2., 2.0], [0., 1.8], [0., 2.3], [1., 4.3]])
# BB = sess.run(flat_concat(C))
# print(BB)
