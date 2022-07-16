#%%
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple
from tensorflow.keras.layers import Lambda


def load_dataset(name, data_dir):
    train1, dataset_info = tfds.load(
        name=name, split="train", data_dir=f"{data_dir}/tfds", with_info=True
    )
    train2 = tfds.load(
        name=name,
        split="validation[100:]",
        data_dir=f"{data_dir}/tfds",
    )
    valid_set = tfds.load(
        name=name,
        split="validation[:100]",
        data_dir=f"{data_dir}/tfds",
    )
    test_set = tfds.load(
        name=name,
        split="train[:10%]",
        data_dir=f"{data_dir}/tfds",
    )
    train_set = train1.concatenate(train2)

    train_num, valid_num, test_num = load_data_num(
        name, data_dir, train_set, valid_set, test_set
    )

    try:
        labels = dataset_info.features["labels"].names
    except:
        labels = dataset_info.features["objects"]["label"].names
    labels = ["bg"] + labels

    return (train_set, valid_set, test_set), labels, train_num, valid_num, test_num


def load_data_num(name, data_dir, train_set, valid_set, test_set):
    data_nums = []
    for dataset, dataset_name in (
        (train_set, "train"),
        (valid_set, "validation"),
        (test_set, "test"),
    ):
        data_num_dir = f"{data_dir}/data_chkr/{''.join(char for char in name if char.isalnum())}_{dataset_name}_num.txt"

        if not (os.path.exists(data_num_dir)):
            data_num = build_data_num(dataset, dataset_name)
            with open(data_num_dir, "w") as f:
                f.write(str(data_num))
                f.close()
        else:
            with open(data_num_dir, "r") as f:
                data_num = int(f.readline())
        data_nums.append(data_num)

    return data_nums


def build_data_num(dataset, dataset_name):
    num_chkr = iter(dataset)
    data_num = 0
    print(f"\nCounting number of {dataset_name} data\n")
    while True:
        try:
            next(num_chkr)
        except:
            break
        data_num += 1

    return data_num


def build_dataset(datasets, batch_size, img_size):
    train_set, valid_set, test_set = datasets
    data_shapes = ([None, None, None], [None, None], [None])
    padding_values = (
        tf.constant(0, tf.float32),
        tf.constant(0, tf.float32),
        tf.constant(-1, tf.int32),
    )

    train_set = train_set.map(lambda x: preprocess(x, split="train", img_size=img_size))
    test_set = test_set.map(lambda x: preprocess(x, split="test", img_size=img_size))
    valid_set = valid_set.map(
        lambda x: preprocess(x, split="validation", img_size=img_size)
    )

    train_set = train_set.repeat().padded_batch(
        batch_size,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    )
    valid_set = valid_set.repeat().batch(1)
    test_set = test_set.repeat().batch(1)

    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)
    valid_set = valid_set.prefetch(tf.data.experimental.AUTOTUNE)
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)

    train_set = iter(train_set)
    valid_set = iter(valid_set)
    test_set = iter(test_set)

    return train_set, valid_set, test_set


def export_data(sample):
    image = Lambda(lambda x: x["image"])(sample)
    gt_boxes = Lambda(lambda x: x["objects"]["bbox"])(sample)
    gt_labels = Lambda(lambda x: x["objects"]["label"])(sample)
    try:
        is_diff = Lambda(lambda x: x["objects"]["is_crowd"])(sample)
    except:
        is_diff = Lambda(lambda x: x["objects"]["is_difficult"])(sample)

    return image, gt_boxes, gt_labels, is_diff


def resize_and_rescale(image, img_size):
    transform = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Resizing(
                img_size[0], img_size[1]
            ),
            tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0),
        ]
    )
    image = transform(image)

    return image


def evaluate(gt_boxes, gt_labels, is_diff):
    not_diff = tf.logical_not(is_diff)
    gt_boxes = Lambda(lambda x: x[not_diff])(gt_boxes)
    gt_labels = Lambda(lambda x: x[not_diff])(gt_labels)

    return gt_boxes, gt_labels


def rand_flip_horiz(image: tf.Tensor, gt_boxes: tf.Tensor) -> Tuple:
    if tf.random.uniform([1]) > tf.constant([0.5]):
        image = tf.image.flip_left_right(image)
        gt_boxes = tf.stack(
            [
                Lambda(lambda x: x[..., 0])(gt_boxes),
                Lambda(lambda x: 1.0 - x[..., 3])(gt_boxes),
                Lambda(lambda x: x[..., 2])(gt_boxes),
                Lambda(lambda x: 1.0 - x[..., 1])(gt_boxes),
            ],
            -1,
        )

    return image, gt_boxes


def preprocess(dataset, split, img_size):
    image, gt_boxes, gt_labels, is_diff = export_data(dataset)
    image = resize_and_rescale(image, img_size)
    if split == "train":
        image, gt_boxes = rand_flip_horiz(image, gt_boxes)
    else:
        gt_boxes, gt_labels = evaluate(gt_boxes, gt_labels, is_diff)
    gt_labels = tf.cast(gt_labels, dtype=tf.int32)

    return image, gt_boxes, gt_labels
