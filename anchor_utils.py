#%%
import tensorflow as tf
from typing import Dict

#%%
def generate_anchors(hyper_params: Dict) -> tf.Tensor:
    """
    generate reference anchors on grid

    Args:
        hyper_params (Dict): hyper parameters

    Returns:
        tf.Tensor: anchors
    """
    feature_map_shape = hyper_params["feature_map_shape"]

    stride = 1 / feature_map_shape

    grid_coords_ctr = tf.cast(
        tf.range(0, feature_map_shape) / feature_map_shape + stride / 2,
        dtype=tf.float32,
    )

    grid_x_ctr, grid_y_ctr = tf.meshgrid(grid_coords_ctr, grid_coords_ctr)

    flat_grid_x_ctr, flat_grid_y_ctr = tf.reshape(grid_x_ctr, (-1,)), tf.reshape(
        grid_y_ctr, (-1,)
    )

    grid_map = tf.stack(
        [flat_grid_y_ctr, flat_grid_x_ctr, flat_grid_y_ctr, flat_grid_x_ctr], axis=-1
    )

    base_anchors = []
    for scale in hyper_params["anchor_scales"]:
        scale /= hyper_params["img_size"]
        for ratio in hyper_params["anchor_ratios"]:
            w = tf.sqrt(scale**2 / ratio)
            h = w * ratio
            base_anchors.append([-h / 2, -w / 2, h / 2, w / 2])
    base_anchors = tf.cast(base_anchors, dtype=tf.float32)

    anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))

    anchors = tf.reshape(anchors, (-1, 4))
    anchors = tf.clip_by_value(t=anchors, clip_value_min=0, clip_value_max=1)

    return anchors
