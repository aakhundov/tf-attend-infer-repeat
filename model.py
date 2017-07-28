import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec

from vae import vae
from transformer import transformer
from gumbel import gumbel_softmax


RESULTS_FOLDER = "air_results/"
DATA_FILE = "multi_mnist_data/common.tfrecords"

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)


EPOCHS = 1000
BATCH_SIZE = 64

CANVAS_SIZE = 50
WINDOW_SIZE = 28

LSTM_UNITS = 256
MAX_DIGITS = 3

VAE_LATENT_DIMENSIONS = 50
VAE_RECOGNITION_UNITS = [512, 256]
VAE_GENERATIVE_UNITS = [256, 512]

ANNEAL_EACH_ITERATIONS = 1000

INIT_Z_PRES_PRIOR = 1e-1
MIN_Z_PRES_PRIOR = 1e-9
Z_PRES_PRIOR_FACTOR = 0.5

INIT_GUMBEL_TEMPERATURE = 10.0
MIN_GUMBEL_TEMPERATURE = 0.1
GUMBEL_TEMPERATURE_FACTOR = 0.8

PLOT_IMAGES_EACH_ITERATIONS = 200
NUMBER_OF_IMAGES_TO_PLOT = 24

SCALE_PRIOR_MEAN = -1.0
SCALE_PRIOR_VARIANCE = 0.1
SHIFT_PRIOR_MEAN = 0.0
SHIFT_PRIOR_VARIANCE = 1.0
VAE_PRIOR_MEAN = 0.0
VAE_PRIOR_VARIANCE = 1.0
VAE_LIKELIHOOD_STD = 0.3

SHIFT_PRIOR_LOG_VARIANCE = tf.log(SHIFT_PRIOR_VARIANCE)
SCALE_PRIOR_LOG_VARIANCE = tf.log(SCALE_PRIOR_VARIANCE)
VAE_PRIOR_LOG_VARIANCE = tf.log(VAE_PRIOR_VARIANCE)

CLIP_GRADIENTS = True
GRADIENT_CLIPPING_NORM = 10.0

LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY = True
LEARNING_RATE_DECAY_RATE = 0.5
LEARNING_RATE_DECAY_STEPS = 1000
LEARNING_RATE_MINIMUM = 1e-4


def read_and_decode(fqueue, batch_size, canvas_size):
    reader = tf.TFRecordReader()
    key, value = reader.read(fqueue)

    features = tf.parse_single_example(
        value,
        features={
            'digits': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            # 'height': tf.FixedLenFeature([], tf.int64),
            # 'width': tf.FixedLenFeature([], tf.int64),
            # 'indices': tf.FixedLenFeature([], tf.string),
            # 'positions': tf.FixedLenFeature([], tf.string),
            # 'boxes': tf.FixedLenFeature([], tf.string),
            # 'labels': tf.FixedLenFeature([], tf.string),
        }
    )

    digs = tf.cast(features['digits'], tf.int32)
    batch = tf.train.shuffle_batch(
        [
            digs,
            tf.reshape(tf.decode_raw(features['image'], tf.float32), [canvas_size * canvas_size]),
            # tf.reshape(tf.pad(tf.decode_raw(features['indices'], tf.int32), [[0, 3 - digs]]), [3]),
            # tf.reshape(tf.pad(tf.decode_raw(features['positions'], tf.int32), [[0, 6 - 2 * digs]]), [6]),
            # tf.reshape(tf.pad(tf.decode_raw(features['boxes'], tf.int32), [[0, 6 - 2 * digs]]), [6]),
            # tf.reshape(tf.pad(tf.decode_raw(features['labels'], tf.int32), [[0, 3 - digs]]), [3])
        ],
        batch_size=batch_size,
        capacity=10000+batch_size*10,
        min_after_dequeue=10000,
    )

    return batch


def sample_from_mvn(mean, diag_variance):
    standard_normal = tf.random_normal(tf.shape(mean))
    sample = mean + standard_normal * tf.sqrt(diag_variance)

    return sample


def plot_digits(original, reconstructed, scales, shifts, digits, iteration):
    num_images = min(NUMBER_OF_IMAGES_TO_PLOT, original.shape[0])

    cols = int(np.sqrt(num_images * 2 / 12) * 4)
    cols = cols if cols % 2 == 0 else cols + 1
    rows = int(np.ceil(num_images * 2 / cols))
    colors = ["r", "g", "b", "m", "w", "y"]

    gs = gridspec.GridSpec(
        rows, cols,
        wspace=0.01, hspace=0.01,
        width_ratios=[1]*cols, height_ratios=[1]*rows,
        top=1.0-0.1/(rows+1), bottom=0.1/(rows+1),
        left=0.1/(cols+1), right=1.0-0.1/(cols+1)
    )

    for i in range(rows):
        for j in range(cols):
            if i*cols + j < num_images*2:
                ax = plt.subplot(gs[i, j])
                img_idx = (i * cols + j) // 2

                if j % 2 == 0:
                    img = original[img_idx]
                else:
                    img = reconstructed[img_idx]

                ax.imshow(
                    np.reshape(img, [CANVAS_SIZE, CANVAS_SIZE]),
                    cmap="gray", vmin=0.0, vmax=1.0
                )

                for d in range(digits[img_idx]):
                    size = CANVAS_SIZE * scales[img_idx][d][0]
                    left = ((CANVAS_SIZE - 1) * (1.0 + shifts[img_idx][d][0]) - size) / 2.0
                    top = ((CANVAS_SIZE - 1) * (1.0 + shifts[img_idx][d][1]) - size) / 2.0

                    ax.add_patch(patches.Rectangle(
                        (left, top), size, size, linewidth=0.5,
                        edgecolor=colors[d], facecolor='none'
                    ))

                ax.axis('off')

    plt.savefig(
        RESULTS_FOLDER + "{0}.png".format(iteration), dpi=600
    )
    plt.clf()


# fetching a batch of numbers of digits and images from a queue
filename_queue = tf.train.string_input_producer([DATA_FILE], num_epochs=EPOCHS)
num_digits, images = read_and_decode(filename_queue, BATCH_SIZE, CANVAS_SIZE)

# creating an RNN cell
cell = rnn.BasicLSTMCell(LSTM_UNITS)
rnn_init_state = cell.zero_state(BATCH_SIZE, images.dtype)

# placeholder for a temperature of Gumbel-Softmax
gumbel_temperature = tf.placeholder(tf.float32, shape=[])
# placeholder for a prior probability of z_pres
z_pres_prior = tf.placeholder(tf.float32, shape=[])


def cond(step, not_finished, *_):
    return tf.logical_and(
        tf.less(step, MAX_DIGITS),
        tf.greater(tf.reduce_max(not_finished), 0.0)
    )


def body(step, not_finished, prev_state, inputs,
         running_recon, running_loss, running_digits,
         scales_ta, shifts_ta):

    # RNN time step
    outputs, next_state = cell(inputs, prev_state)

    # sampling scale
    scale_mean = layers.fully_connected(outputs, 1, activation_fn=None)
    scale_log_variance = layers.fully_connected(outputs, 1, activation_fn=None)
    scale_variance = tf.exp(scale_log_variance)
    scale = tf.nn.sigmoid(sample_from_mvn(scale_mean, scale_variance))
    scales_ta = scales_ta.write(scales_ta.size(), scale)
    s = tf.squeeze(scale)

    # sampling shift
    shift_mean = layers.fully_connected(outputs, 2, activation_fn=None)
    shift_log_variance = layers.fully_connected(outputs, 2, activation_fn=None)
    shift_variance = tf.exp(shift_log_variance)
    shift = tf.nn.tanh(sample_from_mvn(shift_mean, shift_variance))
    shifts_ta = shifts_ta.write(shifts_ta.size(), shift)
    x, y = shift[:, 0], shift[:, 1]

    # ST: theta of forward transformation
    theta = tf.stack([
        tf.concat([tf.stack([s, tf.zeros_like(s)], axis=1), tf.expand_dims(x, 1)], axis=1),
        tf.concat([tf.stack([tf.zeros_like(s), s], axis=1), tf.expand_dims(y, 1)], axis=1),
    ], axis=1)

    # ST forward transformation: canvas -> window
    window = tf.squeeze(transformer(
        tf.expand_dims(tf.reshape(inputs, [-1, CANVAS_SIZE, CANVAS_SIZE]), 3),
        theta, [WINDOW_SIZE, WINDOW_SIZE]
    ))

    # reconstructing the window in VAE
    vae_recon, vae_mean, vae_log_variance = vae(
        tf.reshape(window, [-1, WINDOW_SIZE * WINDOW_SIZE]), WINDOW_SIZE ** 2,
        VAE_RECOGNITION_UNITS, VAE_LATENT_DIMENSIONS, VAE_GENERATIVE_UNITS,
        likelihood_std=VAE_LIKELIHOOD_STD
    )

    # ST: theta of backward transformation
    theta_recon = tf.stack([
        tf.concat([tf.stack([1.0/s, tf.zeros_like(s)], axis=1), tf.expand_dims(-x/s, 1)], axis=1),
        tf.concat([tf.stack([tf.zeros_like(s), 1.0/s], axis=1), tf.expand_dims(-y/s, 1)], axis=1),
    ], axis=1)

    # ST backward transformation: window -> canvas
    window_recon = tf.squeeze(transformer(
        tf.expand_dims(tf.reshape(vae_recon, [-1, WINDOW_SIZE, WINDOW_SIZE]), 3),
        theta_recon, [CANVAS_SIZE, CANVAS_SIZE]
    ))

    # sampling z_pres flag (1 - more digits, 0 - no more digits)
    z_pres_logits = layers.fully_connected(outputs, 2, activation_fn=None)
    z_pres = gumbel_softmax(z_pres_logits, gumbel_temperature, hard=True)[:, 0]
    z_pres_prob = tf.nn.softmax(z_pres_logits)[:, 0]

    # z_pres KL-divergence:
    # previous value of not_finished is used
    # to account for KL of first z_pres=0
    running_loss += not_finished * (
        z_pres_prob * (tf.log(z_pres_prob + 10e-10) - tf.log(z_pres_prior + 10e-10)) +
        (1.0 - z_pres_prob) * (tf.log(1.0 - z_pres_prob + 10e-10) - tf.log(1.0 - z_pres_prior + 10e-10))
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
        tf.reshape(window_recon, [-1, CANVAS_SIZE * CANVAS_SIZE])

    # shift KL-divergence
    running_loss += not_finished * (
        0.5 * tf.reduce_sum(
            SHIFT_PRIOR_LOG_VARIANCE - shift_log_variance - 1.0 + shift_variance / SHIFT_PRIOR_VARIANCE +
            tf.square(shift_mean - SHIFT_PRIOR_MEAN) / SHIFT_PRIOR_VARIANCE, 1
        )
    )

    # scale KL-divergence
    running_loss += not_finished * (
        0.5 * tf.reduce_sum(
            SCALE_PRIOR_LOG_VARIANCE - scale_log_variance - 1.0 + scale_variance / SCALE_PRIOR_VARIANCE +
            tf.square(scale_mean - SCALE_PRIOR_MEAN) / SCALE_PRIOR_VARIANCE, 1
        )
    )

    # VAE KL_DIVERGENCE
    running_loss += not_finished * (
        0.5 * tf.reduce_sum(
            VAE_PRIOR_LOG_VARIANCE - vae_log_variance - 1.0 + tf.exp(vae_log_variance) / VAE_PRIOR_VARIANCE +
            tf.square(vae_mean - VAE_PRIOR_MEAN) / VAE_PRIOR_VARIANCE, 1
        )
    )

    return step+1, not_finished, next_state, inputs, \
        running_recon, running_loss, running_digits, \
        scales_ta, shifts_ta


# RNN while_loop with variable number of steps for each batch item
_, _, _, _, reconstruction, loss, digits, scales, shifts = tf.while_loop(
    cond, body, [
        tf.constant(0), tf.fill([BATCH_SIZE], 1.0),
        rnn_init_state, images, tf.zeros_like(images),
        tf.zeros([BATCH_SIZE]), tf.zeros([BATCH_SIZE], dtype=tf.int32),
        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),
        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    ]
)


# all scales and shifts fetched from while_loop iterations
scales = tf.transpose(scales.stack(), (1, 0, 2))
shifts = tf.transpose(shifts.stack(), (1, 0, 2))

# clipping the reconstructed canvas by [0.0, 1.0]
reconstruction = tf.maximum(tf.minimum(reconstruction, 1.0), 0.0)

# adding reconstruction loss
loss -= tf.reduce_sum(
    images * tf.log(reconstruction + 10e-10) +
    (1.0 - images) * tf.log(1.0 - reconstruction + 10e-10), 1
)

# averaging the loss wrt. a batch
loss = tf.reduce_mean(loss)

# ratio of correctly inferred digit counts
accuracy = tf.reduce_mean(tf.cast(
    tf.equal(num_digits, digits),
    tf.float32
))


# training tools

global_step = tf.Variable(0, trainable=False)

if LEARNING_RATE_DECAY:
    learning_rate = tf.maximum(tf.train.exponential_decay(
        LEARNING_RATE, global_step,
        LEARNING_RATE_DECAY_STEPS,
        LEARNING_RATE_DECAY_RATE,
        staircase=True
    ), LEARNING_RATE_MINIMUM)
else:
    learning_rate = LEARNING_RATE

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads, variables = zip(*optimizer.compute_gradients(loss))

if CLIP_GRADIENTS:
    grads = tf.clip_by_global_norm(grads, GRADIENT_CLIPPING_NORM)[0]

grads_and_vars = list(zip(grads, variables))
train = optimizer.apply_gradients(
    grads_and_vars, global_step=global_step
)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    temp = INIT_GUMBEL_TEMPERATURE
    prob = INIT_Z_PRES_PRIOR

    try:
        while True:
            # training
            _, l, a, iteration = sess.run(
                [train, loss, accuracy, global_step],
                feed_dict={
                    gumbel_temperature: temp,
                    z_pres_prior: prob
                }
            )

            print("iteration {}\tloss {:.3f}\taccuracy {:.2f}".format(iteration, l, a))

            # periodic image saving
            if iteration % PLOT_IMAGES_EACH_ITERATIONS == 0:
                im, rec, sc, sh, dd = sess.run(
                    [images, reconstruction, scales, shifts, digits],
                    feed_dict={
                        gumbel_temperature: temp,
                        z_pres_prior: prob
                    }
                )

                plot_digits(im, rec, sc, sh, dd, iteration)

            # periodic annealing
            if iteration % ANNEAL_EACH_ITERATIONS == 0:
                # annealing Gumbel-Softmax temperature
                temp = max(MIN_GUMBEL_TEMPERATURE, temp * GUMBEL_TEMPERATURE_FACTOR)
                # annealing z_pres prior probability
                prob = max(MIN_Z_PRES_PRIOR, prob * Z_PRES_PRIOR_FACTOR)

                print()
                print("hyperparameters annealed")
                print("temp: {}, prob: {}".format(temp, prob))
                print()

    except tf.errors.OutOfRangeError:
        print()
        print("training has ended")
        print()

    finally:
        coord.request_stop()
        coord.join(threads)
