import os
import shutil

import tensorflow as tf

from multi_mnist import read_and_decode
from multi_mnist import plot_digits
from air_model import AIRModel


RESULTS_FOLDER = "air_results/"
IMAGES_FOLDER = RESULTS_FOLDER + "images/"
MODELS_FOLDER = RESULTS_FOLDER + "models/"
DATA_FILE = "multi_mnist_data/common.tfrecords"

shutil.rmtree(RESULTS_FOLDER, ignore_errors=True)

os.makedirs(RESULTS_FOLDER)
os.makedirs(IMAGES_FOLDER)
os.makedirs(MODELS_FOLDER)


EPOCHS = 1000
BATCH_SIZE = 64
NUM_THREADS = 4

CANVAS_SIZE = 50

SAVE_MODEL_EACH_ITERATIONS = 1000
PLOT_IMAGES_EACH_ITERATIONS = 200
NUMBER_OF_IMAGES_TO_PLOT = 24


# fetching a batch of numbers of digits and images from a queue
filename_queue = tf.train.string_input_producer([DATA_FILE], num_epochs=EPOCHS)
num_digits, images = read_and_decode(filename_queue, BATCH_SIZE, CANVAS_SIZE, NUM_THREADS)

# AIR model
model = AIRModel(
    images, num_digits,
    max_digits=3, lstm_units=256,
    canvas_size=CANVAS_SIZE, windows_size=28,
    vae_latent_dimensions=50, vae_recognition_units=(512, 256), vae_generative_units=(256, 512),
    scale_prior_mean=-1.0, scale_prior_variance=0.1, shift_prior_mean=0.0, shift_prior_variance=1.0,
    vae_prior_mean=0.0, vae_prior_variance=1.0, vae_likelihood_std=0.3,
    z_pres_prior=1e-1, gumbel_temperature=10.0, learning_rate=1e-3,
    gradient_clipping_norm=10.0, annealing_schedules={
        "z_pres_prior": {"init": 1e-1, "min": 1e-9, "factor": 0.5, "iters": 1000},
        "gumbel_temperature": {"init": 10.0, "min": 0.1, "factor": 0.8, "iters": 1000},
        "learning_rate": {"init": 1e-3, "min": 1e-4, "factor": 0.5, "iters": 1000}
    }
)


saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while True:
            # training
            _, l, a, step = sess.run([
                model.training, model.loss,
                model.accuracy, model.global_step
            ])

            print("iteration {}\tloss {:.3f}\taccuracy {:.2f}".format(step, l, a))

            # periodic image plotting
            if step % PLOT_IMAGES_EACH_ITERATIONS == 0:
                im, rec, sc, sh, dd = sess.run([
                    images, model.reconstruction,
                    model.rec_scales, model.rec_shifts,
                    model.rec_num_digits
                ])

                plot_digits(
                    im, rec, sc, sh, dd, step,
                    NUMBER_OF_IMAGES_TO_PLOT,
                    IMAGES_FOLDER
                )

            # periodic model saving
            if step % SAVE_MODEL_EACH_ITERATIONS == 0:
                saver.save(
                    sess, MODELS_FOLDER + "air-model",
                    global_step=model.global_step
                )

    except tf.errors.OutOfRangeError:
        print()
        print("training has ended")
        print()

    finally:
        coord.request_stop()
        coord.join(threads)
