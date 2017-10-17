import os
import math
import shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

from air.air_model import AIRModel
from demo.model_wrapper import ModelWrapper
from multi_mnist import read_test_data


CANVAS_SIZE = 50
WINDOW_SIZE = 28

MNIST_FOLDER = "./mnist_data"
MODEL_PATH = "./model/air-model"
TEST_DATA_FILE = "multi_mnist_data/test.tfrecords"

RESULTS_FOLDER = os.getcwd() + "/embeddings"
SPRITES_FILE = "mnist_sprites.png"
METADATA_FILE = "mnist_metadata.tsv"


def collect_data_info(digits, indices, positions, boxes, labels):
    all_info = []

    for i in range(len(digits)):
        single_info = {
            "num_digits": digits[i],
            "digits": []
        }

        for j in range(digits[i]):
            x, y = positions[i][j * 2:(j + 1) * 2]
            w, h = boxes[i][j * 2:(j + 1) * 2]
            cx = (x + x + w - 1.0) / 2.0
            cy = (y + y + h - 1.0) / 2.0
            st_cx = cx / 24.5 - 1.0
            st_cy = cy / 24.5 - 1.0

            digit_info = {
                "id": indices[i][j],
                "position": [x, y],
                "box": [w, h],
                "center": [cx, cy],
                "st_center": [st_cx, st_cy],
                "label": labels[i][j]
            }

            single_info["digits"].append(digit_info)

        all_info.append(single_info)

    return all_info


def collect_reconstruction_info(digits, positions, windows, latents):
    all_info = []

    for i in range(len(digits)):
        single_info = {
            "num_digits": digits[i],
            "digits": []
        }

        for j in range(digits[i]):
            rec_digit_info = {
                "scale": positions[i][j][0],
                "shift": positions[i][j][1:],
                "window": np.reshape(windows[i][j], (WINDOW_SIZE, WINDOW_SIZE)),
                "latent": latents[i][j]
            }

            single_info["digits"].append(rec_digit_info)

        all_info.append(single_info)

    return all_info


def match_data_with_rec(data_info, reconstruction_info, max_distance=0.1):
    all_info = []

    for img in range(len(data_info)):
        taken = []
        for dig in range(data_info[img]["num_digits"]):
            closest_digit, min_distance = -1, 3.0
            for k in range(reconstruction_info[img]["num_digits"]):
                dist = distance(
                    *data_info[img]["digits"][dig]["st_center"],
                    *reconstruction_info[img]["digits"][k]["shift"]
                )
                if dist < min_distance:
                    min_distance = dist
                    closest_digit = k
            if min_distance <= max_distance and closest_digit not in taken:
                all_info.append({
                    "id": data_info[img]["digits"][dig]["id"],
                    "label": data_info[img]["digits"][dig]["label"],
                    "image": reconstruction_info[img]["digits"][closest_digit]["window"],
                    "latent": reconstruction_info[img]["digits"][closest_digit]["latent"]
                })
                taken.append(closest_digit)

    return all_info


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def create_mnist_sprites(images):
    dim = int(math.ceil(math.sqrt(len(images))))
    sprite_image = np.ones((dim * WINDOW_SIZE, dim * WINDOW_SIZE))

    for i in range(len(images)):
        x, y = i % dim, i // dim
        sprite_image[
            y*WINDOW_SIZE:(y+1)*WINDOW_SIZE,
            x*WINDOW_SIZE:(x+1)*WINDOW_SIZE
        ] -= images[i]

    sprite_file_path = os.path.join(RESULTS_FOLDER, SPRITES_FILE)
    plt.imsave(sprite_file_path, sprite_image, cmap='gray')

    return sprite_file_path


def create_mnist_metadata(labels):
    metadata_file_path = os.path.join(RESULTS_FOLDER, METADATA_FILE)

    with open(metadata_file_path, "w+") as f:
        f.write("Index\tLabel\n")
        for i in range(len(labels)):
            f.write("{0}\t{1}\n".format(i, labels[i]))

    return metadata_file_path


shutil.rmtree(RESULTS_FOLDER, ignore_errors=True)
os.makedirs(RESULTS_FOLDER)


test_data = tf.placeholder(tf.float32, shape=[None, CANVAS_SIZE ** 2])
test_targets = tf.placeholder(tf.int32, shape=[None])

print("Creating model...")
air_model = AIRModel(
    test_data, test_targets,
    max_steps=3, rnn_units=256, canvas_size=CANVAS_SIZE, windows_size=WINDOW_SIZE,
    vae_latent_dimensions=50, vae_recognition_units=(512, 256), vae_generative_units=(256, 512),
    vae_likelihood_std=0.3, scale_hidden_units=64, shift_hidden_units=64, z_pres_hidden_units=64,
    z_pres_temperature=1.0, stopping_threshold=0.99, cnn=False,
    train=False, reuse=False, scope="air",
)


session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True

with tf.Session(config=session_config) as sess:
    print("Restoring model...")
    tf.train.Saver().restore(sess, MODEL_PATH)
    wrapper = ModelWrapper(air_model, sess, test_data)

    print("Reading AIR data...")
    dat_images, dat_digits, dat_indices, dat_positions, dat_boxes, dat_labels = read_test_data(TEST_DATA_FILE)
    dat_info = collect_data_info(dat_digits, dat_indices, dat_positions, dat_boxes, dat_labels)

    print("Reconstructing data...")
    rec_digits, rec_positions, _, rec_windows, rec_latents, _ = wrapper.infer(dat_images)
    rec_info = collect_reconstruction_info(rec_digits, rec_positions, rec_windows, rec_latents)

print("Reading MNIST data...")
mnist = input_data.read_data_sets(MNIST_FOLDER, validation_size=0)

print("Matching original AIR data with reconstructions...")
latent_info = match_data_with_rec(dat_info, rec_info, max_distance=0.2)

label_dic = {d: 0 for d in range(10)}
for info in latent_info:
    label_dic[info["label"]] += 1

print()
print("Present digits: {0}".format(sum(i["num_digits"] for i in dat_info)))
print("Inferred digits: {0}".format(sum(i["num_digits"] for i in rec_info)))
print("Matched inference boxes: {0}".format(len(latent_info)))
print("Digit distribution (among matched digits):")
print(label_dic)
print()

mnist_latents = np.array([info["latent"] for info in latent_info])
mnist_images = np.array([info["image"] for info in latent_info])
mnist_labels = [info["label"] for info in latent_info]

tf.reset_default_graph()

print("Creating embeddings graph...")
embedding_var = tf.Variable(mnist_latents, name="air_mnist")
summary_writer = tf.summary.FileWriter(RESULTS_FOLDER)

projector_config = projector.ProjectorConfig()
embedding = projector_config.embeddings.add()
embedding.tensor_name = embedding_var.name

embedding.metadata_path = create_mnist_metadata(mnist_labels)
embedding.sprite.image_path = create_mnist_sprites(mnist_images)
embedding.sprite.single_image_dim.extend([WINDOW_SIZE, WINDOW_SIZE])
projector.visualize_embeddings(summary_writer, projector_config)

with tf.Session(config=session_config) as sess:
    print("Initializing variables...")
    sess.run(tf.global_variables_initializer())

    print("Saving embeddings...")
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(RESULTS_FOLDER, "embeddings"))
