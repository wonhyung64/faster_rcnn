#%%
import tensorflow as tf
from typing import Dict

#%%
def rpn_reg_loss_fn(
    pred: tf.Tensor, bbox_deltas: tf.Tensor, bbox_labels: tf.Tensor, hyper_params: Dict
) -> tf.Tensor:
    """
    calculate Region Proposal Regression loss

    Args:
        pred (tf.Tensor): RPN reg outputs
        bbox_deltas (tf.Tensor): true offset between anchors and gt_boxes
        bbox_labels (tf.Tensor): objectness of bbox
        hyper_params (Dict): hyper parameters

    Returns:
        tf.Tensor: RPN regression loss
    """
    pred = tf.reshape(
        pred,
        (
            hyper_params["batch_size"],
            hyper_params["feature_map_shape"],
            hyper_params["feature_map_shape"],
            hyper_params["anchor_count"],
            4,
        ),
    )
    bbox_deltas = tf.reshape(
        bbox_deltas,
        (
            hyper_params["batch_size"],
            hyper_params["feature_map_shape"],
            hyper_params["feature_map_shape"],
            hyper_params["anchor_count"],
            4,
        ),
    )

    total_anchors_loc = (
        hyper_params["feature_map_shape"] * hyper_params["feature_map_shape"]
    )

    tune_param = total_anchors_loc / (
        hyper_params["total_pos_bboxes"] + hyper_params["total_neg_bboxes"]
    )

    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)

    loss_for_all = loss_fn(bbox_deltas, pred)

    pos_cond = tf.equal(bbox_labels, tf.constant(1.0))

    pos_mask = tf.cast(pos_cond, dtype=tf.float32)

    loc_loss = tf.reduce_sum(pos_mask * loss_for_all)

    return loc_loss * tune_param / total_anchors_loc


def dtn_reg_loss_fn(
    pred: tf.Tensor,
    frcnn_reg_actuals: tf.Tensor,
    frcnn_cls_actuals: tf.Tensor,
    hyper_params: Dict,
) -> tf.Tensor:
    """
    calculate Detection Network Regression loss

    Args:
        pred (tf.Tensor): detection network regression outputs
        frcnn_reg_actuals (tf.Tensor): true offset between RoI and gt boxes
        frcnn_cls_actuals (tf.Tensor): true class of RoI bbox
        hyper_params (Dict): hyper parameters

    Returns:
        tf.Tensor: detection network regression loss
    """
    pred = tf.reshape(
        pred,
        (
            hyper_params["batch_size"],
            hyper_params["train_nms_topn"],
            hyper_params["total_labels"],
            4,
        ),
    )

    frcnn_reg_actuals = tf.reshape(
        frcnn_reg_actuals,
        (
            hyper_params["batch_size"],
            hyper_params["train_nms_topn"],
            hyper_params["total_labels"],
            4,
        ),
    )

    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)

    loss_for_all = loss_fn(frcnn_reg_actuals, pred)

    pos_cond = tf.equal(frcnn_cls_actuals, tf.constant(1.0))

    pos_mask = tf.cast(pos_cond, dtype=tf.float32)

    loc_loss = tf.reduce_sum(pos_mask * loss_for_all)

    total_pos_bboxes = tf.reduce_sum(pos_mask)

    return tf.clip_by_value(loc_loss, 1e-9, 1e9) / total_pos_bboxes * tf.constant(0.5)


def rpn_cls_loss_fn(pred: tf.Tensor, bbox_labels: tf.Tensor) -> tf.Tensor:
    """
    calculate Region Proposal Classification loss

    Args:
        pred (tf.Tensor): RPN cls outputs
        bbox_labels (tf.Tensor): true objectnetss

    Returns:
        tf.Tensor: RPN cls loss
    """
    indices = tf.where(tf.not_equal(bbox_labels, tf.constant(-1.0, dtype=tf.float32)))

    target = tf.gather_nd(bbox_labels, indices)

    output = tf.gather_nd(pred, indices)

    lf = -tf.reduce_mean(
        target * tf.math.log(output + 1e-7)
        + (1 - target) * tf.math.log(1 - output + 1e-7)
    )
    return lf


def dtn_cls_loss_fn(pred: tf.Tensor, true: tf.Tensor) -> tf.Tensor:
    """
    calculate Detection Network Classification loss

    Args:
        pred (tf.Tensor): detection network cls output
        true (tf.Tensor): true class

    Returns:
        tf.Tensor: detection network classification loss
    """
    loss_for_all = -tf.math.reduce_sum(true * tf.math.log(pred + 1e-7), axis=-1)

    cond = tf.reduce_any(tf.not_equal(true, tf.constant(0.0)), axis=-1)

    mask = tf.cast(cond, dtype=tf.float32)

    conf_loss = tf.reduce_sum(mask * loss_for_all)

    total_boxes = tf.maximum(1.0, tf.reduce_sum(mask))

    return conf_loss / total_boxes
