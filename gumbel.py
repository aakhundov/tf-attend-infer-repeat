import tensorflow as tf


def sample_gumbel(shape, eps=10e-10):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def sample_logistic(count, eps=10e-10):
    U = tf.random_uniform([count], minval=0, maxval=1)
    return tf.log(U + eps) - tf.log(1.0 - U + eps)


def gumbel_softmax_sample_binary(log_odds, temperature):
    y = log_odds + sample_logistic(tf.shape(log_odds)[0])
    return tf.nn.sigmoid(y / temperature)


def gumbel_softmax_binary(log_odds, temperature, hard=False):
    y = gumbel_softmax_sample_binary(log_odds, temperature)
    if hard:
        y_hard = tf.round(y)
        y = tf.stop_gradient(y_hard - y) + y
    return y
