import os

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def crop_bounds(image):
    non_zero_row_ids = np.nonzero(np.sum(image, axis=0))[0]
    non_zero_col_ids = np.nonzero(np.sum(image, axis=1))[0]
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
        print(num_digits, "done")

        if num_digits == MAX_DIGITS_IN_COMMON:
            common_images, common_indices, common_positions, \
              common_boxes, common_labels, common_digits = shuffle_lists(
                common_images, common_indices, common_positions, common_boxes, common_labels, common_digits
              )

            print("Writing 0-{} digit images to common file... ".format(MAX_DIGITS_IN_COMMON), end="", flush=True)
            write_to_records(MULTI_MNIST_FOLDER + "common", common_images, common_indices,
                             common_positions, common_boxes, common_labels, common_digits)
            print("done")
