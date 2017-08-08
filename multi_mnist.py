import os

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec


def crop_bounds(image):
    non_zero_row_ids = np.array(np.nonzero(np.sum(image, axis=0)))[0]
    non_zero_col_ids = np.array(np.nonzero(np.sum(image, axis=1)))[0]
    x_start, x_end = non_zero_row_ids[0], non_zero_row_ids[-1]
    y_start, y_end = non_zero_col_ids[0], non_zero_col_ids[-1]

    return image[y_start:y_end+1, x_start:x_end+1]


def overlaps(canvas, image, x, y):
    h, w = image.shape
    window = canvas[y:y+h, x:x+w]

    return not np.array_equal(np.maximum(image, window), image + window)


def generate_multi_image(single_images, num_images, image_dim, canvas_dim):
    placed_image_ids = []
    placed_image_positions = []
    placed_image_boxes = []

    canvas = np.zeros(
        [canvas_dim, canvas_dim],
        dtype=single_images[0].dtype
    )

    for i in range(num_images):
        placed = False
        while not placed:
            idx = np.random.randint(len(single_images))
            image = np.reshape(single_images[idx], [image_dim, image_dim])
            cropped = crop_bounds(image)
            h, w = cropped.shape

            for attempt in range(10):
                x = np.random.randint(canvas_dim - w)
                y = np.random.randint(canvas_dim - h)

                if not overlaps(canvas, cropped, x, y):
                    canvas[y:y+h, x:x+w] += cropped
                    placed_image_positions.extend([x, y])
                    placed_image_boxes.extend([w, h])
                    placed_image_ids.append(idx)
                    placed = True
                    break

    return canvas, placed_image_ids, placed_image_positions, placed_image_boxes


def write_to_records(filename, images, indices, positions, boxes, labels, digits):
    def _int64_list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    rows, cols = images[0].shape

    writer = tf.python_io.TFRecordWriter(filename + ".tfrecords")
    for index in range(len(images)):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "height": _int64_list_feature([rows]),
                    "width": _int64_list_feature([cols]),
                    "digits": _int64_list_feature([digits[index]]),
                    "indices": _bytes_feature(np.asarray(indices[index], dtype=np.int32).tostring()),
                    "positions": _bytes_feature(np.asarray(positions[index], dtype=np.int32).tostring()),
                    "boxes": _bytes_feature(np.asarray(boxes[index], dtype=np.int32).tostring()),
                    "labels": _bytes_feature(np.asarray(labels[index], dtype=np.int32).tostring()),
                    "image": _bytes_feature(np.ravel(images[index]).tostring())
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()


def shuffle_lists(*lists):
    shuffled = []

    perm = np.random.permutation(
        range(len(lists[0]))
    )

    for l in lists:
        shuffled.append([l[i] for i in perm])

    return shuffled


def read_and_decode(fqueue, batch_size, canvas_size, num_threads):
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
        num_threads=num_threads
    )

    return batch


def plot_digits(original, reconstructed, scales, shifts, digits, iteration, num_images, results_folder):
    num_images = min(num_images, original.shape[0])

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

    canvas_size = int(np.sqrt(original.shape[1]))

    for i in range(rows):
        for j in range(cols):
            if i*cols + j < num_images*2:
                ax = plt.subplot(gs[i, j])
                img_idx = (i * cols + j) // 2

                if j % 2 == 0:
                    image = original[img_idx]
                else:
                    image = reconstructed[img_idx]

                ax.imshow(
                    np.reshape(image, [canvas_size, canvas_size]),
                    cmap="gray", vmin=0.0, vmax=1.0
                )

                for d in range(digits[img_idx]):
                    size = canvas_size * scales[img_idx][d][0]
                    left = ((canvas_size - 1) * (1.0 + shifts[img_idx][d][0]) - size) / 2.0
                    top = ((canvas_size - 1) * (1.0 + shifts[img_idx][d][1]) - size) / 2.0

                    ax.add_patch(patches.Rectangle(
                        (left, top), size, size, linewidth=0.5,
                        edgecolor=colors[d], facecolor='none'
                    ))

                ax.axis('off')

    plt.savefig(
        results_folder + "{0}.png".format(iteration), dpi=600
    )
    plt.clf()


if __name__ == "__main__":

    MAX_DIGITS = 5
    MAX_DIGITS_IN_COMMON = 2
    IMAGES_PER_DIGIT = 20000

    MNIST_FOLDER = "mnist_data/"
    MULTI_MNIST_FOLDER = "multi_mnist_data/"

    if not os.path.exists(MULTI_MNIST_FOLDER):
        os.makedirs(MULTI_MNIST_FOLDER)

    dataset = input_data.read_data_sets(MNIST_FOLDER)

    common_images, common_indices, common_positions = [], [], []
    common_boxes, common_labels, common_digits = [], [], []

    np.random.seed(0)

    print()
    for num_digits in range(MAX_DIGITS+1):
        strata_images, strata_indices = [], []
        strata_positions, strata_boxes = [], []
        strata_labels = []

        print("Generating {} digit images... ".format(num_digits), end="", flush=True)
        for item in range(IMAGES_PER_DIGIT):
            img, ids, pos, box = generate_multi_image(dataset.train.images, num_digits, 28, 50)

            strata_images.append(img)
            strata_indices.append(ids)
            strata_positions.append(pos)
            strata_boxes.append(box)
            strata_labels.append(list(dataset.train.labels[ids]))
        print("done")

        strata_digits = [num_digits] * IMAGES_PER_DIGIT

        if num_digits <= MAX_DIGITS_IN_COMMON:
            common_images.extend(strata_images)
            common_indices.extend(strata_indices)
            common_positions.extend(strata_positions)
            common_boxes.extend(strata_boxes)
            common_labels.extend(strata_labels)
            common_digits.extend(strata_digits)

        print("Writing {} digit images... ".format(num_digits), end="", flush=True)
        write_to_records(MULTI_MNIST_FOLDER + str(num_digits), strata_images, strata_indices,
                         strata_positions, strata_boxes, strata_labels, strata_digits)
        print("done")

        if num_digits == MAX_DIGITS_IN_COMMON:
            common_images, common_indices, common_positions, \
              common_boxes, common_labels, common_digits = shuffle_lists(
                common_images, common_indices, common_positions, common_boxes, common_labels, common_digits
              )

            print("Writing 0-{} digit images to common file... ".format(MAX_DIGITS_IN_COMMON), end="", flush=True)
            write_to_records(MULTI_MNIST_FOLDER + "common", common_images, common_indices,
                             common_positions, common_boxes, common_labels, common_digits)
            print("done")
