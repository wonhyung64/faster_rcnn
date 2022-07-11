import numpy as np
import tensorflow as tf
from .bbox_utils import generate_iou


def calculate_pr(final_bbox, gt_box, mAP_threshold):
    bbox_num = final_bbox.shape[1]
    gt_num = gt_box.shape[1]

    true_pos = tf.Variable(tf.zeros(bbox_num))
    for i in range(bbox_num):
        bbox = tf.split(final_bbox, bbox_num, axis=1)[i]

        iou = generate_iou(bbox, gt_box)

        best_iou = tf.reduce_max(iou, axis=1)
        pos_num = tf.cast(tf.greater(best_iou, mAP_threshold), dtype=tf.float32)
        if tf.reduce_sum(pos_num) >= 1:
            gt_box = gt_box * tf.expand_dims(
                tf.cast(1 - pos_num, dtype=tf.float32), axis=-1
            )
            true_pos = tf.tensor_scatter_nd_update(true_pos, [[i]], [1])
    false_pos = 1.0 - true_pos
    true_pos = tf.math.cumsum(true_pos)
    false_pos = tf.math.cumsum(false_pos)

    recall = true_pos / gt_num
    precision = tf.math.divide(true_pos, true_pos + false_pos)

    return precision, recall


def calculate_ap_per_class(recall, precision):
    interp = tf.constant([i / 10 for i in range(0, 11)])
    AP = tf.reduce_max(
        [tf.where(interp <= recall[i], precision[i], 0.0) for i in range(len(recall))],
        axis=0,
    )
    AP = tf.reduce_sum(AP) / 11
    return AP


def calculate_ap_const(
    final_bboxes, final_labels, gt_boxes, gt_labels, total_labels, mAP_threshold=0.5
):
    AP = []
    for c in range(1, total_labels):
        if tf.math.reduce_any(final_labels == c) or tf.math.reduce_any(gt_labels == c):
            final_bbox = tf.expand_dims(final_bboxes[final_labels == c], axis=0)
            gt_box = tf.expand_dims(gt_boxes[gt_labels == c], axis=0)

            if final_bbox.shape[1] == 0 or gt_box.shape[1] == 0:
                ap = tf.constant(0.0)
            else:
                precision, recall = calculate_pr(final_bbox, gt_box, mAP_threshold)
                ap = calculate_ap_per_class(recall, precision)
            AP.append(ap)
    if AP == []:
        AP = 1.0
    else:
        AP = tf.reduce_mean(AP)
    return AP


def calculate_ap(final_bboxes, final_labels, gt_boxes, gt_labels, total_labels):
    mAP_threshold_lst = np.arange(0.5, 1.0, 0.05)
    APs = []
    for mAP_threshold in mAP_threshold_lst:
        AP = []
        for c in range(1, total_labels):
            if tf.math.reduce_any(final_labels == c) or tf.math.reduce_any(
                gt_labels == c
            ):
                final_bbox = tf.expand_dims(final_bboxes[final_labels == c], axis=0)
                gt_box = tf.expand_dims(gt_boxes[gt_labels == c], axis=0)

                if final_bbox.shape[1] == 0 or gt_box.shape[1] == 0:
                    ap = tf.constant(0.0)
                else:
                    precision, recall = calculate_pr(final_bbox, gt_box, mAP_threshold)
                    ap = calculate_ap_per_class(recall, precision)
                AP.append(ap)
        if AP == []:
            AP = 1.0
        else:
            AP = tf.reduce_mean(AP)
        APs.append(AP)
    return tf.reduce_mean(APs)
