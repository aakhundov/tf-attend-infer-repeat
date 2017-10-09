import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers

from .vae import vae
from .transformer import transformer
from .concrete import concrete_binary_pre_sigmoid_sample
from .concrete import concrete_binary_kl_mc_sample


class AIRModel:

    def __init__(self, input_images, target_num_digits,
                 max_steps=3, max_digits=2, rnn_units=256, canvas_size=50, windows_size=28,
                 vae_latent_dimensions=50, vae_recognition_units=(512, 256), vae_generative_units=(256, 512),
                 scale_prior_mean=-1.0, scale_prior_variance=0.1, shift_prior_mean=0.0, shift_prior_variance=1.0,
                 vae_prior_mean=0.0, vae_prior_variance=1.0, vae_likelihood_std=0.3,
                 scale_hidden_units=64, shift_hidden_units=64, z_pres_hidden_units=64,
                 z_pres_prior_log_odds=-2.0, z_pres_temperature=1.0, stopping_threshold=0.99,
                 learning_rate=1e-3, gradient_clipping_norm=100.0, cnn=True, cnn_filters=8,
                 num_summary_images=60, train=False, reuse=False, scope="air",
                 annealing_schedules=None):

        self.input_images = input_images
        self.target_num_digits = target_num_digits
        self.batch_size = tf.shape(input_images)[0]

        self.max_steps = max_steps
        self.max_digits = max_digits
        self.rnn_units = rnn_units
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

        self.scale_hidden_units = scale_hidden_units
        self.shift_hidden_units = shift_hidden_units
        self.z_pres_hidden_units = z_pres_hidden_units

        self.z_pres_prior_log_odds = z_pres_prior_log_odds
        self.z_pres_temperature = z_pres_temperature
        self.stopping_threshold = stopping_threshold

        self.learning_rate = learning_rate
        self.gradient_clipping_norm = gradient_clipping_norm
        self.num_summary_images = num_summary_images

        self.cnn = cnn
        self.cnn_filters = cnn_filters

        self.train = train

        self.num_summaries = []
        self.img_summaries = []
        self.var_summaries = []
        self.grad_summaries = []

        with tf.variable_scope(scope, reuse=reuse):
            self.global_step = tf.get_variable(name="global_step", shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)

            self.scale_prior_log_variance = tf.log(scale_prior_variance, name="scale_prior_log_variance")
            self.shift_prior_log_variance = tf.log(shift_prior_variance, name="shift_prior_log_variance")
            self.vae_prior_log_variance = tf.log(vae_prior_variance, name="vae_prior_log_variance")

            if annealing_schedules is not None:
                for param, schedule in annealing_schedules.items():
                    # replacing some of the parameters by annealed
                    # versions, if schedule is provided for those
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
    def _create_annealed_tensor(param, schedule, global_step, eps=10e-10):
        value = tf.train.exponential_decay(
            learning_rate=schedule["init"], global_step=global_step,
            decay_steps=schedule["iters"], decay_rate=schedule["factor"],
            staircase=False if "staircase" not in schedule else schedule["staircase"],
            name=param
        )

        if "min" in schedule:
            value = tf.maximum(
                value, schedule["min"],
                name=param + "_max"
            )

        if "max" in schedule:
            value = tf.minimum(
                value, schedule["max"],
                name=param + "_min"
            )

        if "log" in schedule and schedule["log"]:
            value = tf.log(
                value + eps,
                name=param + "_log"
            )

        return value

    @staticmethod
    def _sample_from_mvn(mean, diag_variance):
        # sampling from the multivariate normal
        # with given mean and diagonal covaraince
        standard_normal = tf.random_normal(tf.shape(mean))
        return mean + standard_normal * tf.sqrt(diag_variance)

    @staticmethod
    def _draw_colored_bounding_boxes(images, boxes, steps):
        channels = [images, images, images]

        for s in range(3):
            # empty canvas with s-th bounding box
            step_box = tf.expand_dims(boxes[:, s, :, :], 3)

            for c in range(3):
                if s == c:
                    # adding the box to c-th channel
                    # if the number of attention steps is greater than s
                    channels[c] = tf.where(
                        tf.greater(steps, s),
                        tf.minimum(channels[c] + step_box, tf.ones_like(images)),
                        channels[c]
                    )
                else:
                    # subtracting the box from channels other than c-th
                    # if the number of attention steps is greater than s
                    channels[c] = tf.where(
                        tf.greater(steps, s),
                        tf.maximum(channels[c] - step_box, tf.zeros_like(images)),
                        channels[c]
                    )

        # concatenating all three channels to obtain
        # potentially three R, G, and B bounding boxes
        return tf.concat(channels, axis=3)

    def _summarize_by_digit_count(self, tensor, digits, name):
        # converting to float in case of int tensors
        float_tensor = tf.cast(tensor, tf.float32)

        for i in range(self.max_digits+1):
            # summarizing the scalar for only those
            # images that have exactly i digits
            self.num_summaries.append(
                tf.summary.scalar(
                    name + "_" + str(i) + "_dig",
                    tf.reduce_mean(tf.boolean_mask(
                        float_tensor, tf.equal(digits, i)
                    ))
                )
            )

        # summarizing the scalar for all images
        self.num_summaries.append(
            tf.summary.scalar(
                name + "_all_dig",
                tf.reduce_mean(float_tensor)
            )
        )

    def _summarize_by_step(self, tensor, steps, name, one_more_step=False, all_steps=False):
        # padding (if required) the number of rows in the tensor
        # up to self.max_steps to avoid possible "out of range" errors
        # in case if there were less than self.max_steps steps globally
        tensor = tf.pad(tensor, [[0, 0], [0, self.max_steps - tf.shape(tensor)[1]]])

        for i in range(self.max_steps):
            if all_steps:
                # summarizing the entire (i+1)-st step without
                # differentiating between actual step counts
                self._summarize_by_digit_count(
                    tensor[:, i], self.target_num_digits,
                    name + "_" + str(i+1) + "_step"
                )
            else:
                # considering one more step if required by one_more_step=True
                # by subtracting 1 from loop variable i (e.g. 0 steps > -1)
                mask = tf.greater(steps, i - (1 if one_more_step else 0))

                # summarizing (i+1)-st step only for those
                # batch items that actually had (i+1)-st step
                self._summarize_by_digit_count(
                    tf.boolean_mask(tensor[:, i], mask),
                    tf.boolean_mask(self.target_num_digits, mask),
                    name + "_" + str(i+1) + "_step"
                )

    def _visualize_reconstructions(self, original, reconstruction, st_back, steps, zoom):
        # enlarging the original images
        large_original = tf.image.resize_images(
            tf.reshape(original, [-1, self.canvas_size, self.canvas_size, 1]),
            [zoom * self.canvas_size, zoom * self.canvas_size]
        )

        # enlarging the reconstructions
        large_reconstruction = tf.image.resize_images(
            tf.reshape(reconstruction, [-1, self.canvas_size, self.canvas_size, 1]),
            [zoom * self.canvas_size, zoom * self.canvas_size]
        )

        # padding (if required) the number of backward ST matrices up to
        # self.max_steps to avoid possible misalignment errors in case
        # if there were less than self.max_steps steps globally
        st_back = tf.pad(st_back, [
            [0, 0], [0, self.max_steps - tf.shape(st_back)[1]], [0, 0], [0, 0]
        ])

        # drawing the attention windows
        # using backward ST matrices
        num_images = tf.shape(original)[0]
        boxes = tf.reshape(
            tf.clip_by_value(
                transformer(
                    tf.expand_dims(
                        tf.image.draw_bounding_boxes(
                            tf.zeros(
                                [num_images * self.max_steps, self.windows_size, self.windows_size, 1],
                                dtype=reconstruction.dtype
                            ),
                            tf.tile(
                                [[[0.0, 0.0, 1.0, 1.0]]],
                                [num_images * self.max_steps, 1, 1]
                            )
                        ), 3
                    ), st_back, [zoom * self.canvas_size, zoom * self.canvas_size]
                ), 0.0, 1.0
            ), [num_images, self.max_steps, zoom * self.canvas_size, zoom * self.canvas_size]
        )

        # sharpening the borders
        # of the attention windows
        boxes = tf.where(
            tf.greater(boxes, 0.01),
            tf.ones_like(boxes),
            tf.zeros_like(boxes)
        )

        # concatenating resulting original and reconstructed images with
        # bounding boxes drawn on them and a thin white stripe between them
        return tf.concat([
            self._draw_colored_bounding_boxes(large_original, boxes, steps),         # original images with boxes
            tf.ones([tf.shape(large_original)[0], zoom * self.canvas_size, 4, 3]),   # thin white stripe between
            self._draw_colored_bounding_boxes(large_reconstruction, boxes, steps),   # reconstructed images with boxes
        ], axis=2)

    def _create_model(self):
        # condition of tf.while_loop
        def cond(step, stopping_sum, *_):
            return tf.logical_and(
                tf.less(step, self.max_steps),
                tf.reduce_any(tf.less(stopping_sum, self.stopping_threshold))
            )

        # body of tf.while_loop
        def body(step, stopping_sum, prev_state,
                 running_recon, running_loss, running_digits,
                 scales_ta, shifts_ta, z_pres_probs_ta,
                 z_pres_kls_ta, scale_kls_ta, shift_kls_ta, vae_kls_ta,
                 st_backward_ta):

            with tf.variable_scope("rnn") as scope:
                # RNN time step
                outputs, next_state = cell(self.rnn_input, prev_state, scope=scope)

            with tf.variable_scope("scale"):
                # sampling scale
                with tf.variable_scope("mean"):
                    with tf.variable_scope("hidden") as scope:
                        hidden = layers.fully_connected(outputs, self.scale_hidden_units, scope=scope)
                    with tf.variable_scope("output") as scope:
                        scale_mean = layers.fully_connected(hidden, 1, activation_fn=None, scope=scope)
                with tf.variable_scope("log_variance"):
                    with tf.variable_scope("hidden") as scope:
                        hidden = layers.fully_connected(outputs, self.scale_hidden_units, scope=scope)
                    with tf.variable_scope("output") as scope:
                        scale_log_variance = layers.fully_connected(hidden, 1, activation_fn=None, scope=scope)
                scale_variance = tf.exp(scale_log_variance)
                scale = tf.nn.sigmoid(self._sample_from_mvn(scale_mean, scale_variance))
                scales_ta = scales_ta.write(scales_ta.size(), scale)
                s = tf.squeeze(scale)

            with tf.variable_scope("shift"):
                # sampling shift
                with tf.variable_scope("mean"):
                    with tf.variable_scope("hidden") as scope:
                        hidden = layers.fully_connected(outputs, self.shift_hidden_units, scope=scope)
                    with tf.variable_scope("output") as scope:
                        shift_mean = layers.fully_connected(hidden, 2, activation_fn=None, scope=scope)
                with tf.variable_scope("log_variance"):
                    with tf.variable_scope("hidden") as scope:
                        hidden = layers.fully_connected(outputs, self.shift_hidden_units, scope=scope)
                    with tf.variable_scope("output") as scope:
                        shift_log_variance = layers.fully_connected(hidden, 2, activation_fn=None, scope=scope)
                shift_variance = tf.exp(shift_log_variance)
                shift = tf.nn.tanh(self._sample_from_mvn(shift_mean, shift_variance))
                shifts_ta = shifts_ta.write(shifts_ta.size(), shift)
                x, y = shift[:, 0], shift[:, 1]

            with tf.variable_scope("st_forward"):
                # ST: theta of forward transformation
                theta = tf.stack([
                    tf.concat([tf.stack([s, tf.zeros_like(s)], axis=1), tf.expand_dims(x, 1)], axis=1),
                    tf.concat([tf.stack([tf.zeros_like(s), s], axis=1), tf.expand_dims(y, 1)], axis=1),
                ], axis=1)

                # ST forward transformation: canvas -> window
                window = tf.squeeze(transformer(
                    tf.expand_dims(tf.reshape(self.input_images, [-1, self.canvas_size, self.canvas_size]), 3),
                    theta, [self.windows_size, self.windows_size]
                ))

            with tf.variable_scope("vae"):
                # reconstructing the window in VAE
                vae_recon, vae_mean, vae_log_variance = vae(
                    tf.reshape(window, [-1, self.windows_size * self.windows_size]), self.windows_size ** 2,
                    self.vae_recognition_units, self.vae_latent_dimensions, self.vae_generative_units,
                    self.vae_likelihood_std
                )

            with tf.variable_scope("st_backward"):
                # ST: theta of backward transformation
                theta_recon = tf.stack([
                    tf.concat([tf.stack([1.0 / s, tf.zeros_like(s)], axis=1), tf.expand_dims(-x / s, 1)], axis=1),
                    tf.concat([tf.stack([tf.zeros_like(s), 1.0 / s], axis=1), tf.expand_dims(-y / s, 1)], axis=1),
                ], axis=1)

                # collecting backward transformation matrices of ST
                # to be used for visualizing the attention windows
                st_backward_ta = st_backward_ta.write(st_backward_ta.size(), theta_recon)

                # ST backward transformation: window -> canvas
                window_recon = tf.squeeze(transformer(
                    tf.expand_dims(tf.reshape(vae_recon, [-1, self.windows_size, self.windows_size]), 3),
                    theta_recon, [self.canvas_size, self.canvas_size]
                ))

            with tf.variable_scope("z_pres"):
                # sampling relaxed (continuous) value of z_pres flag
                # from Concrete distribution (closer to 1 - more digits,
                # closer to 0 - no more digits)
                with tf.variable_scope("log_odds"):
                    with tf.variable_scope("hidden") as scope:
                        hidden = layers.fully_connected(outputs, self.z_pres_hidden_units, scope=scope)
                    with tf.variable_scope("output") as scope:
                        z_pres_log_odds = tf.squeeze(
                            layers.fully_connected(hidden, 1, activation_fn=None, scope=scope)
                        )
                with tf.variable_scope("gumbel"):
                    # sampling pre-sigmoid value from concrete distribution
                    # with given location (z_pres_log_odds) and temperature
                    z_pres_pre_sigmoid = concrete_binary_pre_sigmoid_sample(
                        z_pres_log_odds, self.z_pres_temperature
                    )

                    # applying sigmoid to render the Concrete sample
                    z_pres = tf.nn.sigmoid(z_pres_pre_sigmoid)

                    # during test time, rounding the Concrete sample
                    # to obtain the corresponding Bernoulli sample
                    if not self.train:
                        z_pres = tf.round(z_pres)

                    # computing and collecting underlying Bernoulli
                    # probability from inferred log-odds solely for
                    # analysis purposes (not used in the model)
                    z_pres_prob = tf.nn.sigmoid(z_pres_log_odds)
                    z_pres_probs_ta = z_pres_probs_ta.write(z_pres_probs_ta.size(), z_pres_prob)

            with tf.variable_scope("loss/z_pres_kl"):
                # z_pres KL-divergence:
                # previous value of stop_sum is used
                # to account for KL of first z_pres after
                # stop_sum becomes >= 1.0
                z_pres_kl = concrete_binary_kl_mc_sample(
                    z_pres_pre_sigmoid,
                    self.z_pres_prior_log_odds, self.z_pres_temperature,
                    z_pres_log_odds, self.z_pres_temperature
                )

                # adding z_pres KL scaled by z_pres to the loss
                # for those batch items that are not yet finished
                running_loss += tf.where(
                    tf.less(stopping_sum, self.stopping_threshold),
                    z_pres_kl,
                    tf.zeros_like(running_loss)
                )

                # populating z_pres KL's TensorArray with a new value
                z_pres_kls_ta = z_pres_kls_ta.write(z_pres_kls_ta.size(), z_pres_kl)

            # updating stop sum by adding (1 - z_pres) to it:
            # for small z_pres values stop_sum becomes greater
            # or equal to self.stopping_threshold and attention
            # counting of the corresponding batch item stops
            stopping_sum += (1.0 - z_pres)

            # updating inferred number of digits per batch item
            running_digits += tf.cast(tf.less(stopping_sum, self.stopping_threshold), tf.int32)

            with tf.variable_scope("canvas"):
                # adding reconstructed patch scaled
                # by z_pres to the running canvas
                running_recon += tf.where(
                    tf.less(stopping_sum, self.stopping_threshold),
                    tf.expand_dims(z_pres, 1) * tf.reshape(
                        window_recon, [-1, self.canvas_size * self.canvas_size]
                    ),
                    tf.zeros_like(running_recon)
                )

            with tf.variable_scope("loss/scale_kl"):
                # scale KL-divergence
                scale_kl = 0.5 * tf.reduce_sum(
                    self.scale_prior_log_variance - scale_log_variance -
                    1.0 + scale_variance / self.scale_prior_variance +
                    tf.square(scale_mean - self.scale_prior_mean) / self.scale_prior_variance, 1
                )

                # adding scale KL scaled by z_pres to the loss
                # for those batch items that are not yet finished
                running_loss += tf.where(
                    tf.less(stopping_sum, self.stopping_threshold),
                    scale_kl,
                    tf.zeros_like(running_loss)
                )

                # populating scale KL's TensorArray with a new value
                scale_kls_ta = scale_kls_ta.write(scale_kls_ta.size(), scale_kl)

            with tf.variable_scope("loss/shift_kl"):
                # shift KL-divergence
                shift_kl = 0.5 * tf.reduce_sum(
                    self.shift_prior_log_variance - shift_log_variance -
                    1.0 + shift_variance / self.shift_prior_variance +
                    tf.square(shift_mean - self.shift_prior_mean) / self.shift_prior_variance, 1
                )

                # adding shift KL scaled by z_pres to the loss
                # for those batch items that are not yet finished
                running_loss += tf.where(
                    tf.less(stopping_sum, self.stopping_threshold),
                    shift_kl,
                    tf.zeros_like(running_loss)
                )

                # populating shift KL's TensorArray with a new value
                shift_kls_ta = shift_kls_ta.write(shift_kls_ta.size(), shift_kl)

            with tf.variable_scope("loss/VAE_kl"):
                # VAE KL-divergence
                vae_kl = 0.5 * tf.reduce_sum(
                    self.vae_prior_log_variance - vae_log_variance -
                    1.0 + tf.exp(vae_log_variance) / self.vae_prior_variance +
                    tf.square(vae_mean - self.vae_prior_mean) / self.vae_prior_variance, 1
                )

                # adding VAE KL scaled by (1-z_pres) to the loss
                # for those batch items that are not yet finished
                running_loss += tf.where(
                    tf.less(stopping_sum, self.stopping_threshold),
                    vae_kl,
                    tf.zeros_like(running_loss)
                )

                # populating VAE KL's TensorArray with a new value
                vae_kls_ta = vae_kls_ta.write(vae_kls_ta.size(), vae_kl)

            # explicating the shape of "batch-sized"
            # tensors for TensorFlow graph compiler
            stopping_sum.set_shape([None])
            running_digits.set_shape([None])
            running_loss.set_shape([None])

            return step + 1, stopping_sum, next_state, \
                running_recon, running_loss, running_digits, \
                scales_ta, shifts_ta, z_pres_probs_ta, \
                z_pres_kls_ta, scale_kls_ta, shift_kls_ta, vae_kls_ta, \
                st_backward_ta

        if self.cnn:
            with tf.variable_scope("cnn") as cnn_scope:
                cnn_input = tf.reshape(self.input_images, [-1, 50, 50, 1], name="cnn_input")

                conv1 = tf.layers.conv2d(
                    inputs=cnn_input, filters=self.cnn_filters, kernel_size=[5, 5], strides=(1, 1),
                    padding="same", activation=tf.nn.relu, reuse=cnn_scope.reuse, name="conv1"
                )

                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")

                conv2 = tf.layers.conv2d(
                    inputs=pool1, filters=self.cnn_filters, kernel_size=[5, 5], strides=(1, 1),
                    padding="same", activation=tf.nn.relu, reuse=cnn_scope.reuse, name="conv2"
                )

                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool2")

                conv3 = tf.layers.conv2d(
                    inputs=pool2, filters=self.cnn_filters, kernel_size=[5, 5], strides=(1, 1),
                    padding="same", activation=tf.nn.relu, reuse=cnn_scope.reuse, name="conv3"
                )

                self.rnn_input = tf.reshape(conv3, [-1, 12 * 12 * self.cnn_filters], name="cnn_output")
        else:
            self.rnn_input = self.input_images

        with tf.variable_scope("rnn") as rnn_scope:
            # creating RNN cells and initial state
            cell = rnn.GRUCell(self.rnn_units, reuse=rnn_scope.reuse)
            rnn_init_state = cell.zero_state(
                self.batch_size, self.input_images.dtype
            )

            # RNN while_loop with variable number of steps for each batch item
            _, _, _, reconstruction, loss, self.rec_num_digits, scales, shifts, \
                z_pres_probs, z_pres_kls, scale_kls, shift_kls, vae_kls, st_backward = tf.while_loop(
                    cond, body, [
                        tf.constant(0),                                 # RNN time step, initially zero
                        tf.zeros([self.batch_size]),                    # running sum of z_pres samples
                        rnn_init_state,                                 # initial RNN state
                        tf.zeros_like(self.input_images),               # reconstruction canvas, initially empty
                        tf.zeros([self.batch_size]),                    # running value of the loss function
                        tf.zeros([self.batch_size], dtype=tf.int32),    # running inferred number of digits
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # inferred scales
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # inferred shifts
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # z_pres probabilities
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # z_pres KL-divergence
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # scale KL-divergence
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # shift KL-divergence
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # VAE KL-divergence
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)    # backward ST matrices
                    ]
                )

        # transposing contents of TensorArray's fetched from while_loop iterations
        self.rec_scales = tf.transpose(scales.stack(), (1, 0, 2), name="rec_scales")
        self.rec_shifts = tf.transpose(shifts.stack(), (1, 0, 2), name="rec_shifts")
        self.rec_st_back = tf.transpose(st_backward.stack(), (1, 0, 2, 3), name="rec_st_back")
        self.z_pres_probs = tf.transpose(z_pres_probs.stack(), name="z_pres_probs")
        self.z_pres_kls = tf.transpose(z_pres_kls.stack(), name="z_pres_kls")
        self.scale_kls = tf.transpose(scale_kls.stack(), name="scale_kls")
        self.shift_kls = tf.transpose(shift_kls.stack(), name="shift_kls")
        self.vae_kls = tf.transpose(vae_kls.stack(), name="vae_kls")

        with tf.variable_scope("loss/reconstruction"):
            # clipping the reconstructed canvas by [0.0, 1.0]
            self.reconstruction = tf.maximum(tf.minimum(reconstruction, 1.0), 0.0, name="clipped_rec")

            # reconstruction loss: cross-entropy between
            # original images and their reconstructions
            self.reconstruction_loss = -tf.reduce_sum(
                self.input_images * tf.log(self.reconstruction + 10e-10) +
                (1.0 - self.input_images) * tf.log(1.0 - self.reconstruction + 10e-10),
                1, name="reconstruction_loss"
            )

        # adding reconstruction loss
        loss += self.reconstruction_loss

        with tf.variable_scope("accuracy"):
            # accuracy of inferred number of digits
            accuracy = tf.cast(
                tf.equal(self.target_num_digits, self.rec_num_digits),
                tf.float32
            )

        var_scope = tf.get_variable_scope().name
        model_vars = [
            v for v in tf.trainable_variables()
            if v.name.startswith(var_scope)
        ]

        with tf.variable_scope("summaries"):
            # averaging between batch items
            self.loss = tf.reduce_mean(loss, name="loss")
            self.accuracy = tf.reduce_mean(accuracy, name="accuracy")

            # post while-loop numeric summaries grouped by digit count
            self._summarize_by_digit_count(self.rec_num_digits, self.target_num_digits, "steps")
            self._summarize_by_digit_count(self.reconstruction_loss, self.target_num_digits, "rec_loss")
            self._summarize_by_digit_count(accuracy, self.target_num_digits, "digit_acc")
            self._summarize_by_digit_count(loss, self.target_num_digits, "total_loss")

            # step-level numeric summaries (from within while-loop) grouped by step and digit count
            self._summarize_by_step(self.rec_scales[:, :, 0], self.rec_num_digits, "scale")
            self._summarize_by_step(self.z_pres_probs, self.rec_num_digits, "z_pres_prob", all_steps=True)
            self._summarize_by_step(self.z_pres_kls, self.rec_num_digits, "z_pres_kl", one_more_step=True)
            self._summarize_by_step(self.scale_kls, self.rec_num_digits, "scale_kl")
            self._summarize_by_step(self.shift_kls, self.rec_num_digits, "shift_kl")
            self._summarize_by_step(self.vae_kls, self.rec_num_digits, "vae_kl")

            # image summary of the reconstructions
            self.img_summaries.append(
                tf.summary.image(
                    "reconstruction",
                    self._visualize_reconstructions(
                        self.input_images[:self.num_summary_images],
                        self.reconstruction[:self.num_summary_images],
                        self.rec_st_back[:self.num_summary_images],
                        self.rec_num_digits[:self.num_summary_images],
                        zoom=2
                    ),
                    max_outputs=self.num_summary_images
                )
            )

            # variable summaries
            for v in model_vars:
                self.var_summaries.append(
                    tf.summary.histogram(
                        v.name,
                        v.value()
                    )
                )

        if self.train:
            with tf.variable_scope("training"):
                # optimizer to minimize the loss function
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                grads, variables = zip(*optimizer.compute_gradients(self.loss))

                if self.gradient_clipping_norm is not None:
                    for i in range(len(grads)):
                        if grads[i] is None:
                            continue
                        # summaries of the original gradient values
                        self.grad_summaries.append(tf.summary.histogram(
                            variables[i].name + "_grad_original", grads[i]
                        ))
                        self.grad_summaries.append(tf.summary.scalar(
                            variables[i].name + "_grad_original_norm", tf.norm(grads[i])
                        ))
                        self.grad_summaries.append(tf.summary.scalar(
                            variables[i].name + "_grad_original_avg", tf.reduce_mean(grads[i])
                        ))

                    # gradient clipping by global norm, if required
                    grads = tf.clip_by_global_norm(grads, self.gradient_clipping_norm)[0]

                for i in range(len(grads)):
                    if grads[i] is None:
                        continue
                    # summaries of the applied (maybe clipped) gradients
                    self.grad_summaries.append(tf.summary.histogram(
                        variables[i].name + "_grad_applied", grads[i]
                    ))
                    self.grad_summaries.append(tf.summary.scalar(
                        variables[i].name + "_grad_applied_norm", tf.norm(grads[i])
                    ))
                    self.grad_summaries.append(tf.summary.scalar(
                        variables[i].name + "_grad_applied_avg", tf.reduce_mean(grads[i])
                    ))

                grads_and_vars = list(zip(grads, variables))

                # training step operation
                self.training = optimizer.apply_gradients(
                    grads_and_vars, global_step=self.global_step
                )
