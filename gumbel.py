import tensorflow as tf


def sample_logistic(count, eps=10e-10):
    u = tf.random_uniform([count], minval=0, maxval=1)
    return tf.log(u + eps) - tf.log(1.0 - u + eps)


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
