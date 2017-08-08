import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers

from vae import vae
from transformer import transformer
from gumbel import gumbel_softmax_binary


class AIRModel:

    def __init__(self, input_images, target_num_digits,
                 max_steps=3, max_digits=2, lstm_units=256, canvas_size=50, windows_size=28,
                 vae_latent_dimensions=50, vae_recognition_units=(512, 256), vae_generative_units=(256, 512),
                 scale_prior_mean=-1.0, scale_prior_variance=0.1, shift_prior_mean=0.0, shift_prior_variance=1.0,
                 vae_prior_mean=0.0, vae_prior_variance=1.0, vae_likelihood_std=0.3,
                 z_pres_prior=0.01, gumbel_temperature=1.0,
                 learning_rate=1e-4, gradient_clipping_norm=10.0,
                 annealing_schedules=None):

        self.input_images = input_images
        self.target_num_digits = target_num_digits
        self.batch_size = tf.shape(input_images)[0]

        self.max_steps = max_steps
        self.max_digits = max_digits
        self.lstm_units = lstm_units
        self.canvas_size = canvas_size
        self.windows_size = windows_size

        self.vae_latent_dimensions = vae_latent_dimensions
        self.vae_recognition_units = vae_recognition_units
        self.vae_generative_units = vae_generative_units

        self.scale_prior_mean = scale_prior_mean
        self.scale_prior_variance = scale_prior_variance
        self.shift_prior_mean = shift_prior_mean
        self.shift_prior_variance = shift_prior_variance
        self.vae_prior_mean = vae_prior_mean
        self.vae_prior_variance = vae_prior_variance
        self.vae_likelihood_std = vae_likelihood_std

        self.z_pres_prior = z_pres_prior
        self.gumbel_temperature = gumbel_temperature
        self.learning_rate = learning_rate
        self.gradient_clipping_norm = gradient_clipping_norm

        with tf.name_scope("air"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step")

            self.scale_prior_log_variance = tf.log(scale_prior_variance, name="scale_prior_log_variance")
            self.shift_prior_log_variance = tf.log(shift_prior_variance, name="shift_prior_log_variance")
            self.vae_prior_log_variance = tf.log(vae_prior_variance, name="vae_prior_log_variance")

            if annealing_schedules is not None:
                for param, schedule in annealing_schedules.items():
                    setattr(self, param, self._create_annealed_tensor(
                        param, schedule, self.global_step
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
    def _create_annealed_tensor(param, schedule, global_step):
        return tf.maximum(
            tf.train.exponential_decay(
                schedule["init"], global_step,
                schedule["iters"], schedule["factor"],
                staircase=True, name=param
            ),
            schedule["min"],
            name=param + "_max"
        )

    @staticmethod
    def _sample_from_mvn(mean, diag_variance):
        standard_normal = tf.random_normal(tf.shape(mean))
        return mean + standard_normal * tf.sqrt(diag_variance)

    def _summary_by_digit_count(self, tensor, digits, name):
        float_tensor = tf.cast(tensor, tf.float32)

        for i in range(self.max_digits+1):
            tf.summary.scalar(
                name + "_" + str(i) + "_dig",
                tf.reduce_mean(tf.boolean_mask(
                    float_tensor, tf.equal(digits, i)
                )), ["num.summaries"]
            )

        tf.summary.scalar(
            name + "_all_dig",
            tf.reduce_mean(float_tensor),
            ["num.summaries"]
        )

    def _summary_by_step_and_digit_count(self, tensor, steps, name):
        for i in range(self.max_steps):
            mask = tf.greater(steps, i)
            self._summary_by_digit_count(
                tf.boolean_mask(tensor[:, i], mask),
                tf.boolean_mask(self.target_num_digits, mask),
                name + "_" + str(i+1) + "_step"
            )

        mask = tf.greater(steps, 0)
        masked_sum = tf.reduce_sum(
            tensor * tf.cast(tf.sequence_mask(steps, self.max_steps), tensor.dtype),
            axis=1
        )

        self._summary_by_digit_count(
            tf.boolean_mask(masked_sum / tf.cast(steps, tensor.dtype), mask),
            tf.boolean_mask(self.target_num_digits, mask),
            name + "_avg_step"
        )

    def _create_model(self):
        # condition of tf.while_loop
        def cond(step, not_finished, *_):
            return tf.logical_and(
                tf.less(step, self.max_steps),
                tf.greater(tf.reduce_max(not_finished), 0.0)
            )

        # body of tf.while_loop
        def body(step, not_finished, prev_state, inputs,
                 running_recon, running_loss, running_digits,
                 scales_ta, shifts_ta, z_pres_probs_ta,
                 z_pres_kls_ta, scale_kls_ta, shift_kls_ta, vae_kls_ta):

            with tf.name_scope("lstm") as scope:
                # RNN time step
                outputs, next_state = cell(inputs, prev_state, scope=scope)

            with tf.name_scope("scale"):
                # sampling scale
                with tf.name_scope("mean") as scope:
                    scale_mean = layers.fully_connected(outputs, 1, activation_fn=None, scope=scope)
                with tf.name_scope("log_variance") as scope:
                    scale_log_variance = layers.fully_connected(outputs, 1, activation_fn=None, scope=scope)
                scale_variance = tf.exp(scale_log_variance)
                scale = tf.nn.sigmoid(self._sample_from_mvn(scale_mean, scale_variance))
                scales_ta = scales_ta.write(scales_ta.size(), scale)
                s = tf.squeeze(scale)

            with tf.name_scope("shift"):
                # sampling shift
                with tf.name_scope("mean") as scope:
                    shift_mean = layers.fully_connected(outputs, 2, activation_fn=None, scope=scope)
                with tf.name_scope("log_variance") as scope:
                    shift_log_variance = layers.fully_connected(outputs, 2, activation_fn=None, scope=scope)
                shift_variance = tf.exp(shift_log_variance)
                shift = tf.nn.tanh(self._sample_from_mvn(shift_mean, shift_variance))
                shifts_ta = shifts_ta.write(shifts_ta.size(), shift)
                x, y = shift[:, 0], shift[:, 1]

            with tf.name_scope("st_forward"):
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

            with tf.name_scope("vae"):
                # reconstructing the window in VAE
                vae_recon, vae_mean, vae_log_variance = vae(
                    tf.reshape(window, [-1, self.windows_size * self.windows_size]), self.windows_size ** 2,
                    self.vae_recognition_units, self.vae_latent_dimensions, self.vae_generative_units,
                    self.vae_likelihood_std
                )

            with tf.name_scope("st_backward"):
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

            with tf.name_scope("z_pres"):
                # sampling z_pres flag (1 - more digits, 0 - no more digits)
                with tf.name_scope("log_odds") as scope:
                    z_pres_log_odds = tf.squeeze(layers.fully_connected(outputs, 1, activation_fn=None, scope=scope))
                with tf.name_scope("gumbel"):
                    z_pres = gumbel_softmax_binary(z_pres_log_odds, self.gumbel_temperature, hard=True)
                    z_pres_prob = tf.exp(z_pres_log_odds) / (1.0 + tf.exp(z_pres_log_odds))
                    z_pres_probs_ta = z_pres_probs_ta.write(z_pres_probs_ta.size(), z_pres_prob)

            with tf.name_scope("loss/z_pres_kl"):
                # z_pres KL-divergence:
                # previous value of not_finished is used
                # to account for KL of first z_pres=0
                z_pres_kl = not_finished * (
                    z_pres_prob * (
                        tf.log(z_pres_prob + 10e-10) -
                        tf.log(self.z_pres_prior + 10e-10)
                    ) +
                    (1.0 - z_pres_prob) * (
                        tf.log(1.0 - z_pres_prob + 10e-10) -
                        tf.log(1.0 - self.z_pres_prior + 10e-10)
                    )
                )
                z_pres_kls_ta = z_pres_kls_ta.write(z_pres_kls_ta.size(), z_pres_kl)
                running_loss += z_pres_kl

            # updating finishing status
            not_finished = tf.where(
                tf.equal(not_finished, 1.0),
                z_pres * tf.stop_gradient(not_finished),
                tf.zeros_like(not_finished)
            )

            # running inferred number of digits per batch item
            running_digits += tf.cast(not_finished, tf.int32)

            with tf.name_scope("canvas"):
                # adding reconstructed window to the running canvas
                running_recon += tf.expand_dims(not_finished, 1) * \
                    tf.reshape(window_recon, [-1, self.canvas_size * self.canvas_size])

            with tf.name_scope("loss/scale_kl"):
                # scale KL-divergence
                scale_kl = not_finished * (
                    0.5 * tf.reduce_sum(
                        self.scale_prior_log_variance - scale_log_variance -
                        1.0 + scale_variance / self.scale_prior_variance +
                        tf.square(scale_mean - self.scale_prior_mean) / self.scale_prior_variance, 1
                    )
                )
                scale_kls_ta = scale_kls_ta.write(scale_kls_ta.size(), scale_kl)
                running_loss += scale_kl

            with tf.name_scope("loss/shift_kl"):
                # shift KL-divergence
                shift_kl = not_finished * (
                    0.5 * tf.reduce_sum(
                        self.shift_prior_log_variance - shift_log_variance -
                        1.0 + shift_variance / self.scale_prior_variance +
                        tf.square(shift_mean - self.shift_prior_mean) / self.scale_prior_variance, 1
                    )
                )
                shift_kls_ta = shift_kls_ta.write(shift_kls_ta.size(), shift_kl)
                running_loss += shift_kl

            with tf.name_scope("loss/VAE_kl"):
                # VAE KL_DIVERGENCE
                vae_kl = not_finished * (
                    0.5 * tf.reduce_sum(
                        self.vae_prior_log_variance - vae_log_variance -
                        1.0 + tf.exp(vae_log_variance) / self.vae_prior_variance +
                        tf.square(vae_mean - self.vae_prior_mean) / self.vae_prior_variance, 1
                    )
                )
                vae_kls_ta = vae_kls_ta.write(vae_kls_ta.size(), vae_kl)
                running_loss += vae_kl

            return step + 1, not_finished, next_state, inputs, \
                running_recon, running_loss, running_digits, \
                scales_ta, shifts_ta, z_pres_probs_ta, \
                z_pres_kls_ta, scale_kls_ta, shift_kls_ta, vae_kls_ta

        with tf.name_scope("rnn"):
            # creating RNN cells and initial state
            cell = rnn.BasicLSTMCell(self.lstm_units)
            rnn_init_state = cell.zero_state(
                self.batch_size, self.input_images.dtype
            )

            # RNN while_loop with variable number of steps for each batch item
            _, _, _, _, reconstruction, loss, self.rec_num_digits, scales, shifts, \
                z_pres_probs, z_pres_kls, scale_kls, shift_kls, vae_kls = tf.while_loop(
                    cond, body, [
                        tf.constant(0),                                 # RNN time step, initially zero
                        tf.fill([self.batch_size], 1.0),                # "not_finished" tensor, 1 - not finished
                        rnn_init_state,                                 # initial RNN state
                        self.input_images,                              # original images go to each RNN step
                        tf.zeros_like(self.input_images),               # reconstruction canvas, initially empty
                        tf.zeros([self.batch_size]),                    # running value of the loss function
                        tf.zeros([self.batch_size], dtype=tf.int32),    # running inferred number of digits
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # inferred scales
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # inferred shifts
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # z_pres probabilities
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # z_pres KL-divergence
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # scale KL-divergence
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # shift KL-divergence
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)    # VAE KL-divergence
                    ]
                )

        # transposing contents of TensorArray's fetched from while_loop iterations
        self.rec_scales = tf.transpose(scales.stack(), (1, 0, 2), name="rec_scales")
        self.rec_shifts = tf.transpose(shifts.stack(), (1, 0, 2), name="rec_shifts")
        self.z_pres_probs = tf.transpose(z_pres_probs.stack(), name="z_pres_probs")
        self.z_pres_kls = tf.transpose(z_pres_kls.stack(), name="z_pres_kls")
        self.scale_kls = tf.transpose(scale_kls.stack(), name="scale_kls")
        self.shift_kls = tf.transpose(shift_kls.stack(), name="shift_kls")
        self.vae_kls = tf.transpose(vae_kls.stack(), name="vae_kls")

        with tf.name_scope("loss/reconstruction"):
            # clipping the reconstructed canvas by [0.0, 1.0]
            self.reconstruction = tf.maximum(tf.minimum(reconstruction, 1.0), 0.0, name="clipped_rec")

            # reconstruction loss: cross-entropy between
            # original images and their reconstructions
            self.reconstruction_loss = -tf.reduce_sum(
                self.input_images * tf.log(self.reconstruction + 10e-10) +
                (1.0 - self.input_images) * tf.log(1.0 - self.reconstruction + 10e-10),
                1, name="reconstruction_loss"
            )

            loss += self.reconstruction_loss

        # scalar summaries grouped by digit count
        self._summary_by_digit_count(self.rec_num_digits, self.target_num_digits, "steps")
        self._summary_by_digit_count(self.reconstruction_loss, self.target_num_digits, "rec_loss")
        self._summary_by_digit_count(loss, self.target_num_digits, "total_loss")

        # step-level summaries group by step and digit count
        self._summary_by_step_and_digit_count(self.rec_scales[:, :, 0], self.rec_num_digits, "scale")
        self._summary_by_step_and_digit_count(self.z_pres_probs, self.rec_num_digits, "z_pres_prob")
        self._summary_by_step_and_digit_count(self.z_pres_kls, self.rec_num_digits, "z_pres_kl")
        self._summary_by_step_and_digit_count(self.scale_kls, self.rec_num_digits, "scale_kl")
        self._summary_by_step_and_digit_count(self.shift_kls, self.rec_num_digits, "shift_kl")
        self._summary_by_step_and_digit_count(self.vae_kls, self.rec_num_digits, "vae_kl")

        # averaging the loss wrt. batch items
        self.loss = tf.reduce_mean(loss, name="loss")

        with tf.name_scope("accuracy"):
            # accuracy of inferred number of digits
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(self.target_num_digits, self.rec_num_digits),
                tf.float32
            ))

        with tf.name_scope("training"):
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
