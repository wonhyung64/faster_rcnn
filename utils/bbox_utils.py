#%%
import tensorflow as tf

#%%
def generate_base_anchors(hyper_params):
    img_size = hyper_params["img_size"]
    anchor_ratios = hyper_params["anchor_ratios"]
    anchor_scales = hyper_params["anchor_scales"]
    base_anchors = []
    for scale in anchor_scales:
        scale /= img_size
        for ratio in anchor_ratios:
            w = tf.sqrt(scale **2 / ratio)
            h = w * ratio
            base_anchors.append([-h / 2, -w / 2, h / 2, w / 2])
    return tf.cast(base_anchors, dtype=tf.float32)

#%%
def generate_anchors(hyper_params):
    anchor_count = hyper_params["anchor_count"]
    feature_map_shape = hyper_params["feature_map_shape"]
    
    stride = 1 / feature_map_shape
    grid_coords = tf.cast(tf.range(0, feature_map_shape) / feature_map_shape + stride / 2, dtype = tf.float32)
    
    grid_x, grid_y = tf.meshgrid(grid_coords, grid_coords)
    flat_grid_x, flat_grid_y = tf.reshape(grid_x, (-1, )), tf.reshape(grid_y, (-1, ))
    grid_map = tf.stack([flat_grid_y, flat_grid_x, flat_grid_y, flat_grid_x], axis = -1)
    
    base_anchors = generate_base_anchors(hyper_params)
    
    anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))
    anchors = tf.reshape(anchors, (-1, 4))
    return tf.clip_by_value(anchors, 0, 1)

#%%
def non_max_suppression(pred_bboxes, pred_labels, **kwargs):
    return tf.image.combined_non_max_suppression(
        pred_bboxes,
        pred_labels,
        *kwargs
    )
# %%
def get_bboxes_from_deltas(anchors, deltas):
    all_anc_width = anchors[..., 3] - anchors[..., 1]
    all_anc_height = anchors[..., 2] - anchors[..., 0]
    all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height
    
    all_bbox_width = tf.exp(deltas[..., 3]) * all_anc_width
    all_bbox_height = tf.exp(deltas[..., 2]) * all_anc_height
    all_bbox_ctr_x = (deltas[..., 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (deltas[..., 0] * all_anc_height) + all_anc_ctr_y
    
    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1
    
    return tf.stack([y1, x1, y2, x2], axis = -1)

#%%
def get_deltas_from_bboxes(bboxes, gt_boxes):
    bbox_width = bboxes[..., 3] - bboxes[..., 1]
    bbox_height = bboxes[..., 2] - bboxes[..., 0]
    bbox_ctr_x = bboxes[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[..., 0] + 0.5 * bbox_height
    
    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height
    
    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / bbox_height))
    delta_h = tf.where(tf.equal(gt_height,0), tf.zeros_like(gt_height), tf.math.log(gt_height / bbox_height))
    
    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)

#%%
# def generate_iou_map(bboxes, gt_boxes):
    
