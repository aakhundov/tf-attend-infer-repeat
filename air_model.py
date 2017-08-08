import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers

from vae import vae
from transformer import transformer
from gumbel import gumbel_softmax_binary


class AIRModel:

    def __init__(self, input_images, target_num_digits,
                 max_digits=3, lstm_units=256, canvas_size=50, windows_size=28,
                 vae_latent_dimensions=50, vae_recognition_units=(512, 256), vae_generative_units=(256, 512),
                 scale_prior_mean=-1.0, scale_prior_variance=0.1, shift_prior_mean=0.0, shift_prior_variance=1.0,
                 vae_prior_mean=0.0, vae_prior_variance=1.0, vae_likelihood_std=0.3,
                 z_pres_prior=0.01, gumbel_temperature=1.0,
                 learning_rate=1e-4, gradient_clipping_norm=10.0,
                 annealing_schedules=None):

        self.input_images = input_images
        self.target_num_digits = target_num_digits
        self.batch_size = tf.shape(input_images)[0]

        self.max_digits = max_digits
        self.lstm_units = lstm_units
        self.canvas_size = canvas_size
        self.windows_size = windows_size

        self.vae_latent_dimensions = vae_latent_dimensions
        self.vae_recognition_units = vae_recognition_units
        self.vae_generative_units = vae_generative_units

        self.scale_prior_mean = scale_prior_mean
        self.scale_prior_variance = scale_prior_variance
        self.scale_prior_log_variance = tf.log(scale_prior_variance)
        self.shift_prior_mean = shift_prior_mean
        self.shift_prior_variance = shift_prior_variance
        self.shift_prior_log_variance = tf.log(shift_prior_variance)
        self.vae_prior_mean = vae_prior_mean
        self.vae_prior_variance = vae_prior_variance
        self.vae_prior_log_variance = tf.log(vae_prior_variance)
        self.vae_likelihood_std = vae_likelihood_std

        self.z_pres_prior = z_pres_prior
        self.gumbel_temperature = gumbel_temperature
        self.learning_rate = learning_rate
        self.gradient_clipping_norm = gradient_clipping_norm

        self.global_step = tf.Variable(0, trainable=False)

        if annealing_schedules is not None:
            for param, schedule in annealing_schedules.items():
                setattr(self, param, self._create_annealed_tensor(
                    schedule, self.global_step
                ))

        self.rec_num_digits = None
        self.rec_scales = None
        self.rec_shifts = None
        self.reconstruction = None
        self.loss = None
        self.accuracy = None
        self.training = None

        self._create_model()

    @staticmethod
    def _create_annealed_tensor(schedule, global_step):
        return tf.maximum(
            tf.train.exponential_decay(
                schedule["init"], global_step,
                schedule["iters"], schedule["factor"],
                staircase=True
            ),
            schedule["min"]
        )

    @staticmethod
    def _sample_from_mvn(mean, diag_variance):
        standard_normal = tf.random_normal(tf.shape(mean))
        return mean + standard_normal * tf.sqrt(diag_variance)

    def _create_model(self):
        # creating RNN cells and initial state
        cell = rnn.BasicLSTMCell(self.lstm_units)
        rnn_init_state = cell.zero_state(
            self.batch_size, self.input_images.dtype
        )

        # condition of tf.while_loop
        def cond(step, not_finished, *_):
            return tf.logical_and(
                tf.less(step, self.max_digits),
                tf.greater(tf.reduce_max(not_finished), 0.0)
            )

        # body of tf.while_loop
        def body(step, not_finished, prev_state, inputs,
                 running_recon, running_loss, running_digits,
                 scales_ta, shifts_ta):

            # RNN time step
            outputs, next_state = cell(inputs, prev_state)

            # sampling scale
            scale_mean = layers.fully_connected(outputs, 1, activation_fn=None)
            scale_log_variance = layers.fully_connected(outputs, 1, activation_fn=None)
            scale_variance = tf.exp(scale_log_variance)
            scale = tf.nn.sigmoid(self._sample_from_mvn(scale_mean, scale_variance))
            scales_ta = scales_ta.write(scales_ta.size(), scale)
            s = tf.squeeze(scale)

            # sampling shift
            shift_mean = layers.fully_connected(outputs, 2, activation_fn=None)
            shift_log_variance = layers.fully_connected(outputs, 2, activation_fn=None)
            shift_variance = tf.exp(shift_log_variance)
            shift = tf.nn.tanh(self._sample_from_mvn(shift_mean, shift_variance))
            shifts_ta = shifts_ta.write(shifts_ta.size(), shift)
            x, y = shift[:, 0], shift[:, 1]

            # ST: theta of forward transformation
            theta = tf.stack([
                tf.concat([tf.stack([s, tf.zeros_like(s)], axis=1), tf.expand_dims(x, 1)], axis=1),
                tf.concat([tf.stack([tf.zeros_like(s), s], axis=1), tf.expand_dims(y, 1)], axis=1),
            ], axis=1)

            # ST forward transformation: canvas -> window
            window = tf.squeeze(transformer(
                tf.expand_dims(tf.reshape(inputs, [-1, self.canvas_size, self.canvas_size]), 3),
                theta, [self.windows_size, self.windows_size]
            ))

            # reconstructing the window in VAE
            vae_recon, vae_mean, vae_log_variance = vae(
                tf.reshape(window, [-1, self.windows_size * self.windows_size]), self.windows_size ** 2,
                self.vae_recognition_units, self.vae_latent_dimensions, self.vae_generative_units,
                self.vae_likelihood_std
            )

            # ST: theta of backward transformation
            theta_recon = tf.stack([
                tf.concat([tf.stack([1.0 / s, tf.zeros_like(s)], axis=1), tf.expand_dims(-x / s, 1)], axis=1),
                tf.concat([tf.stack([tf.zeros_like(s), 1.0 / s], axis=1), tf.expand_dims(-y / s, 1)], axis=1),
            ], axis=1)

            # ST backward transformation: window -> canvas
            window_recon = tf.squeeze(transformer(
                tf.expand_dims(tf.reshape(vae_recon, [-1, self.windows_size, self.windows_size]), 3),
                theta_recon, [self.canvas_size, self.canvas_size]
            ))

            # sampling z_pres flag (1 - more digits, 0 - no more digits)
            z_pres_log_odds = tf.squeeze(layers.fully_connected(outputs, 1, activation_fn=None))
            z_pres = gumbel_softmax_binary(z_pres_log_odds, self.gumbel_temperature, hard=True)
            z_pres_prob = tf.exp(z_pres_log_odds) / (1.0 + tf.exp(z_pres_log_odds))

            # z_pres KL-divergence:
            # previous value of not_finished is used
            # to account for KL of first z_pres=0
            running_loss += not_finished * (
                z_pres_prob * (tf.log(z_pres_prob + 10e-10) - tf.log(self.z_pres_prior + 10e-10)) +
                (1.0 - z_pres_prob) * (tf.log(1.0 - z_pres_prob + 10e-10) - tf.log(1.0 - self.z_pres_prior + 10e-10))
            )

            # updating finishing status
            not_finished = tf.where(
                tf.equal(not_finished, 1.0),
                z_pres * tf.stop_gradient(not_finished),
                tf.zeros_like(not_finished)
            )

            # number of digits per batch item
            running_digits += tf.cast(not_finished, tf.int32)

            # adding reconstructed window to the running canvas
            running_recon += tf.expand_dims(not_finished, 1) * \
                tf.reshape(window_recon, [-1, self.canvas_size * self.canvas_size])

            # shift KL-divergence
            running_loss += not_finished * (
                0.5 * tf.reduce_sum(
                    self.shift_prior_log_variance - shift_log_variance -
                    1.0 + shift_variance / self.scale_prior_variance +
                    tf.square(shift_mean - self.shift_prior_mean) / self.scale_prior_variance, 1
                )
            )

            # scale KL-divergence
            running_loss += not_finished * (
                0.5 * tf.reduce_sum(
                    self.scale_prior_log_variance - scale_log_variance -
                    1.0 + scale_variance / self.scale_prior_variance +
                    tf.square(scale_mean - self.scale_prior_mean) / self.scale_prior_variance, 1
                )
            )

            # VAE KL_DIVERGENCE
            running_loss += not_finished * (
                0.5 * tf.reduce_sum(
                    self.vae_prior_log_variance - vae_log_variance -
                    1.0 + tf.exp(vae_log_variance) / self.vae_prior_variance +
                    tf.square(vae_mean - self.vae_prior_mean) / self.vae_prior_variance, 1
                )
            )

            return step + 1, not_finished, next_state, inputs, \
                running_recon, running_loss, running_digits, \
                scales_ta, shifts_ta

        # RNN while_loop with variable number of steps for each batch item
        _, _, _, _, reconstruction, loss, digits, scales, shifts = tf.while_loop(
            cond, body, [
                tf.constant(0),                                 # RNN time step, initially zero
                tf.fill([self.batch_size], 1.0),                # "not_finished" tensor, 1 - not finished
                rnn_init_state,                                 # initial RNN state
                self.input_images,                              # original images go to each RNN step
                tf.zeros_like(self.input_images),               # reconstruction canvas, initially empty
                tf.zeros([self.batch_size]),                    # running value of the loss function
                tf.zeros([self.batch_size], dtype=tf.int32),    # running number of inferred digits
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # inferred scales
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)    # inferred shifts
            ]
        )

        # the inferred digit count
        self.rec_num_digits = digits

        # all scales and shifts fetched from while_loop iterations
        self.rec_scales = tf.transpose(scales.stack(), (1, 0, 2))
        self.rec_shifts = tf.transpose(shifts.stack(), (1, 0, 2))

        # clipping the reconstructed canvas by [0.0, 1.0]
        self.reconstruction = tf.maximum(tf.minimum(reconstruction, 1.0), 0.0)

        # adding reconstruction loss
        loss -= tf.reduce_sum(
            self.input_images * tf.log(self.reconstruction + 10e-10) +
            (1.0 - self.input_images) * tf.log(1.0 - self.reconstruction + 10e-10), 1
        )

        # averaging the loss wrt. a batch
        self.loss = tf.reduce_mean(loss)

        # accuracy of inferred number of digits
        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(self.target_num_digits, self.rec_num_digits),
            tf.float32
        ))

        # optimizer to minimize the loss function
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads, variables = zip(*optimizer.compute_gradients(self.loss))

        if self.gradient_clipping_norm is not None:
            # gradient clipping by global norm, if required
            grads = tf.clip_by_global_norm(grads, self.gradient_clipping_norm)[0]

        grads_and_vars = list(zip(grads, variables))

        # training step operation
        self.training = optimizer.apply_gradients(
            grads_and_vars, global_step=self.global_step
        )
