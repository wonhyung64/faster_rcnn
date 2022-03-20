#%%
import numpy as np
import tensorflow as tf
from typing import Tuple

#%%
def preprocessing(img: tf.Tensor, gt_boxes: tf.Tensor, gt_labels: tf.Tensor) -> Tuple:
    """
    preprocess tensors with augmentations

    Args:
        img (tf.Tensor): input images
        gt_boxes (tf.Tensor): input gt boxes
        gt_labels (tf.Tensor): input gt labels

    Returns:
        Tuple: tuple of preprocessed tensors
    """
    gt_labels = tf.cast(gt_labels + 1, tf.int32)
    if (np.random.uniform(0, 1, 1) > 0.5) == True:
        img, gt_boxes = flip_horizontal(img, gt_boxes)

    return img, gt_boxes, gt_labels


def preprocessing_ship(
    img: tf.Tensor, gt_boxes: tf.Tensor, gt_labels: tf.Tensor, filename: str = None
) -> tf.Tensor:
    """
    preprocess tensors with augmentations for ship data

    Args:
        img (tf.Tensor): input images
        gt_boxes (tf.Tensor): input gt boxes
        gt_labels (tf.Tensor): input gt labels
        filename (str, optional): filenames of input. Defaults to None.

    Returns:
        tf.Tensor: tuple of preprocessed tensors
    """
    gt_labels = tf.cast(gt_labels + 1, tf.int32)
    if (np.random.uniform(0, 1, 1) > 0.5) == True:
        img, gt_boxes = flip_horizontal(img, gt_boxes)

    return img, gt_boxes, gt_labels


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
