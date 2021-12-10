#%%
import numpy as np
import tensorflow as tf
#%%
def calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params, chk_pos_num):
    batch_size = hyper_params['batch_size'] 
    feature_map_shape = hyper_params['feature_map_shape'] 
    anchor_count = hyper_params['anchor_count']  
    total_pos_bboxes = hyper_params['total_pos_bboxes'] 
    total_neg_bboxes = hyper_params['total_neg_bboxes']
    variances = hyper_params['variances'] 
    pos_threshold = hyper_params["pos_threshold"]
    neg_threshold = hyper_params["neg_threshold"]

    iou_map = generate_iou(anchors, gt_boxes)
    #
    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    merged_iou_map = tf.reduce_max(iou_map, axis=2) 

    pos_mask = tf.greater(merged_iou_map, pos_threshold)
    chk_pos_num = np.size(np.where(pos_mask[0] == True))
    #
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    #

    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)
    max_pos_mask = tf.scatter_nd(indices=scatter_bbox_indices, updates=tf.fill((tf.shape(valid_indices)[0], ), True), shape=tf.shape(pos_mask))
    pos_mask = tf.logical_or(pos_mask, max_pos_mask)
    pos_mask = randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
    #
    pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1)
    neg_count = (total_pos_bboxes + total_neg_bboxes) - pos_count
    
    neg_mask = tf.logical_and(tf.less(merged_iou_map, neg_threshold), tf.logical_not(pos_mask))
    neg_mask = randomly_select_xyz_mask(neg_mask, neg_count)
    #
    
    pos_labels = tf.where(pos_mask, tf.ones_like(pos_mask, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32))
    
    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)
    gt_boxes_map = tf.gather(params=gt_boxes, indices=max_indices_each_row, batch_dims=1)

    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    
    bbox_deltas = bbox_to_delta(anchors, expanded_gt_boxes) / variances

    bbox_deltas = tf.reshape(bbox_deltas, (batch_size, feature_map_shape, feature_map_shape, anchor_count* 4))
    bbox_labels = tf.reshape(bbox_labels, (batch_size, feature_map_shape, feature_map_shape, anchor_count))
    
    return bbox_deltas, bbox_labels, chk_pos_num
    
#%%
def randomly_select_xyz_mask(mask, select_xyz):
    maxval = tf.reduce_max(select_xyz) * 10
    random_mask = tf.random.uniform(tf.shape(mask), minval=1, maxval=maxval, dtype=tf.int32)
    multiplied_mask = tf.cast(mask, tf.int32) * random_mask
    sorted_mask = tf.argsort(multiplied_mask, direction="DESCENDING")
    sorted_mask_indices = tf.argsort(sorted_mask)
    selected_mask = tf.less(sorted_mask_indices, tf.expand_dims(select_xyz, 1))
    return tf.logical_and(mask, selected_mask)
    
#%%
def generate_iou(anchors, gt_boxes):
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(anchors, 4, axis=-1) 
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1) 
    
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1) 
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis = -1)
    
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1])) 
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))
    
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area)
    
    return intersection_area / union_area 
#%%
def bbox_to_delta(anchors, gt_boxes):
    bbox_width = anchors[..., 3] - anchors[..., 1]
    bbox_height = anchors[..., 2] - anchors[...,0]
    bbox_ctr_x = anchors[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = anchors[..., 0] + 0.5 * bbox_height
    
    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height
    
    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / bbox_width))
    delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.math.log(gt_height / bbox_height))
    
    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)