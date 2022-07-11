import sys
import subprocess
import tensorflow as tf

from .loss_utils import (
    rpn_reg_loss_fn,
    rpn_cls_loss_fn,
    dtn_reg_loss_fn,
    dtn_cls_loss_fn,
)


def build_optimizer(batch_size, data_num):
    boundaries = [data_num // batch_size * epoch for epoch in (10, 60, 90)]
    values = [1e-5, 1e-6, 1e-7, 1e-8]
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )

    optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_fn)
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr_fn)

    return optimizer1, optimizer2


@tf.function
def forward_backward_rpn(image, true, model, optimizer, batch_size, feature_map_shape, anchor_ratios, anchor_scales, total_pos_bboxes, total_neg_bboxes):
    bbox_deltas, bbox_labels = true
    with tf.GradientTape(persistent=True) as tape:
        """RPN"""
        rpn_reg_output, rpn_cls_output, feature_map = model(image)

        rpn_reg_loss = rpn_reg_loss_fn(
            rpn_reg_output,
            bbox_deltas,
            bbox_labels,
            batch_size,
            feature_map_shape,
            anchor_ratios,
            anchor_scales,
            total_pos_bboxes,
            total_neg_bboxes,
            )
        rpn_cls_loss = rpn_cls_loss_fn(rpn_cls_output, bbox_labels)
        rpn_loss = rpn_reg_loss + rpn_cls_loss

    grads_rpn = tape.gradient(rpn_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads_rpn, model.trainable_weights))

    return (rpn_reg_loss, rpn_cls_loss), rpn_reg_output, rpn_cls_output, feature_map


@tf.function
def forward_backward_dtn(pooled_roi, true, model, optimizer, total_labels, batch_size, train_nms_topn):
    roi_deltas, roi_labels = true
    with tf.GradientTape(persistent=True) as tape:
        """DTN"""
        dtn_reg_output, dtn_cls_output = model(pooled_roi, training=True)

        dtn_reg_loss = dtn_reg_loss_fn(
            dtn_reg_output,
            roi_deltas,
            roi_labels,
            total_labels,
            batch_size,
            train_nms_topn
            )
        dtn_cls_loss = dtn_cls_loss_fn(dtn_cls_output, roi_labels)
        dtn_loss = dtn_reg_loss + dtn_cls_loss

    grads_dtn = tape.gradient(dtn_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads_dtn, model.trainable_weights))

    return (dtn_reg_loss, dtn_cls_loss)
