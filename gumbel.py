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


def concrete_binary_sample(log_odds, temperature, hard=False, eps=10e-10):
    count = tf.shape(log_odds)[0]

    u = tf.random_uniform([count], minval=0, maxval=1)
    noise = tf.log(u + eps) - tf.log(1.0 - u + eps)

    y = log_odds + noise
    sig_y = tf.nn.sigmoid(y / temperature)

    if hard:
        sig_y_hard = tf.round(sig_y)
        sig_y = tf.stop_gradient(sig_y_hard - sig_y) + sig_y

    return y, sig_y


def concrete_binary_kl_mc(y, log_odds1, temperature1, log_odds2, temperature2, eps=10e-10):
    temp1_y, temp2_y = temperature1 * y, temperature2 * y

    log1 = tf.log(temperature1 + eps) - temp1_y + log_odds1 - 2.0 * tf.log(1.0 + tf.exp(-temp1_y + log_odds1) + eps)
    log2 = tf.log(temperature2 + eps) - temp2_y + log_odds2 - 2.0 * tf.log(1.0 + tf.exp(-temp2_y + log_odds2) + eps)

    return log1 - log2
