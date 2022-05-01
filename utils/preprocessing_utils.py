import numpy as np
import tensorflow as tf
from typing import Tuple

def preprocessing(dataset: tf.data.Dataset, img_size, eval: bool = False) -> Tuple:
    """
    preprocess tensors with augmentations

    Args:
        dataset (tf.data.Dataset): input images
        eval (bool): option for evaluation

    Returns:
        Tuple: tuple of preprocessed tensors
    """
    image = dataset["image"]/255
    image = tf.image.resize(image, img_size)
    bbox = dataset["objects"]["bbox"]
    label = dataset["objects"]["label"]
    if eval == "True":
        not_diff = tf.logical_not(dataset["objects"]["is_difficult"])
        bbox = bbox[not_diff]
        label = label[not_diff]
    label = tf.cast(label + 1, tf.int32)
    if (np.random.uniform(0, 1, 1) > 0.5) == True:
        image, bbox = flip_horizontal(image, bbox)

    return image, bbox, label


def flip_horizontal(img: tf.Tensor, gt_boxes: tf.Tensor) -> Tuple:
    """
    flip image and gt boxes horizontally

    Args:
        img (tf.Tensor): input images
        gt_boxes (tf.Tensor): input gt boxes

    Returns:
        Tuple: tuple of flipped images and gt boxes
    """
    img = tf.image.flip_left_right(img)
    gt_boxes = tf.stack(
        [
            gt_boxes[..., 0],
            1.0 - gt_boxes[..., 3],
            gt_boxes[..., 2],
            1.0 - gt_boxes[..., 1],
        ],
        -1,
    )

    return img, gt_boxes
