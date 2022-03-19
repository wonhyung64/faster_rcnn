#%%
import tensorflow as tf

#%%
def delta_to_bbox(anchors: tf.Tensor, bbox_deltas: tf.Tensor) -> tf.Tensor:
    """
    transform bbox offset to coordinates

    Args:
        anchors (tf.Tensor): reference anchors
        bbox_deltas (tf.Tensor): bbox offset

    Returns
        tf.Tensor: bbox coordinates
    """
    all_anc_width = anchors[..., 3] - anchors[..., 1]
    all_anc_height = anchors[..., 2] - anchors[..., 0]
    all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height

    all_bbox_width = tf.exp(bbox_deltas[..., 3]) * all_anc_width
    all_bbox_height = tf.exp(bbox_deltas[..., 2]) * all_anc_height
    all_bbox_ctr_x = (bbox_deltas[..., 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (bbox_deltas[..., 0] * all_anc_height) + all_anc_ctr_y

    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1

    return tf.stack([y1, x1, y2, x2], axis=-1)


#%%
def bbox_to_delta(anchors: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """
    transform bbox coordinates to offset

    Args:
        anchors (tf.Tensor): reference anchors
        gt_boxes (tf.Tensor): bbox to transform

    Returns:
        tf.Tensor: bbox offset
    """
    bbox_width = anchors[..., 3] - anchors[..., 1]
    bbox_height = anchors[..., 2] - anchors[..., 0]
    bbox_ctr_x = anchors[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = anchors[..., 0] + 0.5 * bbox_height

    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height

    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    delta_x = tf.where(
        tf.equal(gt_width, 0),
        tf.zeros_like(gt_width),
        tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width),
    )
    delta_y = tf.where(
        tf.equal(gt_height, 0),
        tf.zeros_like(gt_height),
        tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height),
    )
    delta_w = tf.where(
        tf.equal(gt_width, 0),
        tf.zeros_like(gt_width),
        tf.math.log(gt_width / bbox_width),
    )
    delta_h = tf.where(
        tf.equal(gt_height, 0),
        tf.zeros_like(gt_height),
        tf.math.log(gt_height / bbox_height),
    )

    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)


#%%
def generate_iou(anchors: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """
    calculate Intersection over Union

    Args:
        anchors (tf.Tensor): reference anchors
        gt_boxes (tf.Tensor): bbox to calculate IoU

    Returns:
        tf.Tensor: Intersection over Union
    """
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(anchors, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)

    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)

    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1]))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))

    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(
        y_bottom - y_top, 0
    )

    union_area = (
        tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area
    )

    return intersection_area / union_area
