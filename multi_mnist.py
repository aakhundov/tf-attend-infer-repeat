import os
import argparse

import numpy as np
import scipy.ndimage as nd
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def show_image(image):
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    plt.show()


def read_image(path, max_intensity):
    image = nd.imread(path, mode="L")
    image = np.asarray(image, dtype=np.float32) / 255.0
    img_min, img_max = image.min(), image.max()

    if img_min != img_max:
        if img_min > 0.0:
            image -= img_min
        if img_max > 0.0:
            image /= img_max
        if max_intensity < 1.0:
            image *= max_intensity
    else:
        if img_max > max_intensity:
            image = np.ones_like(image) * max_intensity

    return image


def crop_non_empty(image):
    non_zero_row_ids = np.array(np.nonzero(np.sum(image, axis=0)))[0]
    non_zero_col_ids = np.array(np.nonzero(np.sum(image, axis=1)))[0]
    x_start, x_end = non_zero_row_ids[0], non_zero_row_ids[-1]
    y_start, y_end = non_zero_col_ids[0], non_zero_col_ids[-1]

    return image[y_start:y_end+1, x_start:x_end+1]


def add_buffer(image, buffer_width):
    b = buffer_width
    w, h = image.shape
    result = np.copy(image)

    for x in range(w):
        for y in range(h):
            if image[y, x] > 0:
                for i in range(x-b, x+b+1):
                    for j in range(y-b, y+b+1):
                        if (0 <= i < w and 0 <= j < h) and result[j, i] == 0:
                            result[j, i] = 1.0

    return result


def pixels_overlap(canvas, image, x, y):
    h, w = image.shape
    window = canvas[y:y+h, x:x+w]

    return not np.array_equal(np.maximum(image, window), image + window)


def bounding_boxes_overlap(x, y, w, h, positions, boxes, gap):
    for i in range(len(positions) // 2):
        p, b = positions[i*2:(i+1)*2], boxes[i*2:(i+1)*2]
        l1x, l1y, r1x, r1y = x-gap, y-gap, x+w+gap-1, y+h+gap-1
        l2x, l2y, r2x, r2y = p[0], p[1], p[0]+b[0]-1, p[1]+b[1]-1

        if l1x <= r2x and l2x <= r1x:
            return True
        if l1y >= r2y and l2y >= r1y:
            return True

    return False


def generate_multi_image(single_images, num_images, image_dim, canvas_dim, bg=None,
                         min_w=1.0, max_w=1.0, min_h=1.0, max_h=1.0, min_ang=0.0, max_ang=0.0,
                         gap=0, margin=0, use_pixel_overlap=False):

    global digit_ids, next_digit_id, used_digit_ids

    ready = False
    while not ready:
        canvas = np.zeros(
            [canvas_dim, canvas_dim],
            dtype=single_images[0].dtype
        )

        placed_image_ids = []
        placed_image_positions = []
        placed_image_boxes = []

        if num_digits == 0:
            break

        try:
            for i in range(num_images):
                idx = digit_ids[next_digit_id]
                next_digit_id += 1
                if next_digit_id >= len(digit_ids):
                    digit_ids = np.random.permutation(digit_ids)
                    next_digit_id = 0

                image = np.reshape(single_images[idx], [image_dim, image_dim])
                image = crop_non_empty(image)

                if min_w != 1.0 or max_w != 1.0 or min_h != 1.0 or max_h != 1.0:
                    new_width = np.random.uniform(min_w, max_w)
                    new_height = np.random.uniform(min_h, max_h)

                    image = nd.affine_transform(
                        image,
                        matrix=np.array([[1.0 / new_height, 0.0], [0.0, 1.0 / new_width]]),
                        output_shape=(int(image_dim * new_height), int(image_dim * new_width)),
                        order=5
                    )

                    image = np.clip(image, 0.0, 1.0)
                    image = np.where(image >= 0.05, image, np.zeros_like(image))
                    image = crop_non_empty(image)

                if min_ang != 0.0 or max_ang != 0.0:
                    new_angle = np.random.uniform(min_ang, max_ang)

                    image = nd.rotate(image, new_angle, order=5)

                    image = np.clip(image, 0.0, 1.0)
                    image = np.where(image >= 0.05, image, np.zeros_like(image))
                    image = crop_non_empty(image)

                h, w = image.shape
                position_find_attempts = 0
                position_found = False

                while position_find_attempts < 100:
                    x = np.random.randint(margin, canvas_dim - w - margin + 1)
                    y = np.random.randint(margin, canvas_dim - h - margin + 1)

                    if i == 0:
                        position_found = True
                    else:
                        if use_pixel_overlap:
                            position_found = not pixels_overlap(
                                canvas_with_buffer, image, x, y
                            )
                        else:
                            position_found = not bounding_boxes_overlap(
                                x, y, w, h, placed_image_positions, placed_image_boxes, gap
                            )

                    if position_found:
                        break

                    position_find_attempts += 1

                if position_found:
                    canvas[y:y+h, x:x+w] += image

                    if use_pixel_overlap and num_digits > 1:
                        canvas_with_buffer = add_buffer(canvas, gap) if gap > 0 else canvas

                    placed_image_positions.extend([x, y])
                    placed_image_boxes.extend([w, h])
                    placed_image_ids.append(idx)

                    if i == num_images - 1:
                        ready = True
                else:
                    break

        except IndexError:
            pass

    if bg is not None:
        canvas = np.clip(canvas + bg, 0.0, 1.0)

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
        shuffled.append(np.array([l[i] for i in perm]))

    return shuffled


def read_and_decode(fqueue, batch_size, canvas_size, num_threads):
    reader = tf.TFRecordReader()
    key, value = reader.read(fqueue)

    features = tf.parse_single_example(
        value,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'digits': tf.FixedLenFeature([], tf.int64)
        }
    )

    batch = tf.train.shuffle_batch(
        [
            tf.reshape(tf.decode_raw(features['image'], tf.float32), [canvas_size * canvas_size]),
            tf.cast(features['digits'], tf.int32)
        ],
        batch_size=batch_size,
        capacity=10000+batch_size*10,
        min_after_dequeue=10000,
        num_threads=num_threads
    )

    return batch


def read_test_data(filename):
    record_iterator = tf.python_io.tf_record_iterator(path=filename)

    images_list, digits_list = [], []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        images_list.append(np.fromstring(example.features.feature['image'].bytes_list.value[0], dtype=np.float32))
        digits_list.append(int(example.features.feature['digits'].int64_list.value[0]))

    empty = [i for i in range(len(digits_list)) if digits_list[i] == 0]
    non_empty = [i for i in range(len(digits_list)) if digits_list[i] > 0]

    images_list, digits_list = np.array(images_list), np.array(digits_list)
    images_list = np.concatenate([np.array([images_list[empty[0]]]), images_list[non_empty], images_list[empty[1:]]])
    digits_list = np.concatenate([np.array([digits_list[empty[0]]]), digits_list[non_empty], digits_list[empty[1:]]])

    return images_list, digits_list


if __name__ == "__main__":

    DEFAULT_MAX_DIGITS = 2
    DEFAULT_MAX_IN_COMMON = 2
    DEFAULT_IMAGES_PER_DIGIT = 20000
    DEFAULT_TEST_SET_SIZE = 1000

    MNIST_FOLDER = "mnist_data/"
    MULTI_MNIST_FOLDER = "multi_mnist_data/"

    CANVAS_SIZE = 50
    IMAGE_SIZE = 28

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-digits", type=int, choices=list(range(7)), default=DEFAULT_MAX_DIGITS)
    parser.add_argument("--max-in-common", type=int, choices=list(range(7)), default=DEFAULT_MAX_IN_COMMON)
    parser.add_argument("--images-per-digit", type=int, default=DEFAULT_IMAGES_PER_DIGIT)
    parser.add_argument("--test-set-size", type=int, default=DEFAULT_TEST_SET_SIZE)
    parser.add_argument("--digit-gap", type=int, default=0)
    parser.add_argument("--canvas-margin", type=int, default=0)
    parser.add_argument("--bg-path", default="")
    parser.add_argument("--bg-max-intensity", type=float, default=1.0)
    parser.add_argument("--min-width-scale", type=float, default=1.0)
    parser.add_argument("--max-width-scale", type=float, default=1.0)
    parser.add_argument("--min-height-scale", type=float, default=1.0)
    parser.add_argument("--max-height-scale", type=float, default=1.0)
    parser.add_argument("--min-rotation-angle", type=float, default=0.0)
    parser.add_argument("--max-rotation-angle", type=float, default=0.0)
    parser.add_argument("--use-pixel-overlap", action='store_true')
    parser.set_defaults(use_pixel_overlap=False)
    args = parser.parse_args()

    if not os.path.exists(MULTI_MNIST_FOLDER):
        os.makedirs(MULTI_MNIST_FOLDER)

    background = read_image(args.bg_path, args.bg_max_intensity) if args.bg_path != "" else None

    dataset = input_data.read_data_sets(MNIST_FOLDER, validation_size=0)

    common_images, common_indices, common_positions = [], [], []
    common_boxes, common_labels, common_digits = [], [], []

    np.random.seed(0)

    next_digit_id = 0
    used_digit_ids = set([])
    digit_ids = [i for i in range(len(dataset.train.images))]
    digit_ids = np.random.permutation(digit_ids)

    for num_digits in range(args.max_digits + 1):
        strata_images, strata_indices = [], []
        strata_positions, strata_boxes = [], []
        strata_labels = []

        print()
        print("Generating {} digit images...".format(num_digits))
        for item in range(args.images_per_digit):
            img, ids, pos, box = generate_multi_image(
                dataset.train.images, num_digits, IMAGE_SIZE, CANVAS_SIZE, bg=background,
                min_w=args.min_width_scale, max_w=args.max_width_scale,
                min_h=args.min_height_scale, max_h=args.max_height_scale,
                min_ang=args.min_rotation_angle, max_ang=args.max_rotation_angle,
                gap=args.digit_gap, margin=args.canvas_margin,
                use_pixel_overlap=args.use_pixel_overlap
            )

            if num_digits <= args.max_in_common:
                for digit_id in ids:
                    used_digit_ids.add(digit_id)

            strata_images.append(img)
            strata_indices.append(ids)
            strata_positions.append(pos)
            strata_boxes.append(box)
            strata_labels.append(list(dataset.train.labels[ids]))

            if (item + 1) % 1000 == 0:
                print("{0} done".format(item + 1))
        print()

        strata_digits = [num_digits] * args.images_per_digit

        if num_digits <= args.max_in_common:
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

        if num_digits == args.max_in_common:
            common_images, common_indices, common_positions, \
              common_boxes, common_labels, common_digits = shuffle_lists(
                common_images, common_indices, common_positions, common_boxes, common_labels, common_digits
              )

            print()
            print("{0} MNIST digits used for 0-{1} digit images".format(len(used_digit_ids), args.max_in_common))
            print("Writing 0-{} digit images to common file... ".format(args.max_in_common), end="", flush=True)
            write_to_records(MULTI_MNIST_FOLDER + "common",
                             common_images[args.test_set_size:], common_indices[args.test_set_size:],
                             common_positions[args.test_set_size:], common_boxes[args.test_set_size:],
                             common_labels[args.test_set_size:], common_digits[args.test_set_size:])
            print("done")

            print("Writing 0-{} digit images to test file... ".format(args.max_in_common), end="", flush=True)
            write_to_records(MULTI_MNIST_FOLDER + "test",
                             common_images[:args.test_set_size], common_indices[:args.test_set_size],
                             common_positions[:args.test_set_size], common_boxes[:args.test_set_size],
                             common_labels[:args.test_set_size], common_digits[:args.test_set_size])
            print("done")
