import tensorflow as tf
import tensorflow.contrib.layers as layers


def vae(inputs, input_dim, rec_hidden_units, latent_dim, gen_hidden_units, likelihood_std=0.0, activation=tf.nn.softplus):

    input_size = tf.shape(inputs)[0]

    next_layer = inputs
    for units in rec_hidden_units:
        next_layer = layers.fully_connected(next_layer, units, activation_fn=activation)

    recognition_mean = layers.fully_connected(next_layer, latent_dim, activation_fn=None)
    recognition_log_variance = layers.fully_connected(next_layer, latent_dim, activation_fn=None)
    standard_normal_sample = tf.random_normal([input_size, latent_dim])

    recognition_sample = recognition_mean + standard_normal_sample * tf.sqrt(tf.exp(recognition_log_variance))

    next_layer = recognition_sample
    for units in gen_hidden_units:
        next_layer = layers.fully_connected(next_layer, units, activation_fn=activation)

    generative_mean = layers.fully_connected(next_layer, input_dim, activation_fn=None)
    standard_normal_sample2 = tf.random_normal([input_size, input_dim])

    generative_sample = generative_mean + standard_normal_sample2 * likelihood_std

    reconstruction = tf.nn.sigmoid(
        generative_sample
    )

    return reconstruction, recognition_mean, recognition_log_variance
