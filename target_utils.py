#%%
import numpy as np
import tensorflow as tf
import bbox_utils
#%%
def rpn_target(anchors, gt_boxes, gt_labels, hyper_params):
    batch_size = hyper_params['batch_size'] 
    feature_map_shape = hyper_params['feature_map_shape'] 
    anchor_count = hyper_params['anchor_count']  
    total_pos_bboxes = hyper_params['total_pos_bboxes'] 
    total_neg_bboxes = hyper_params['total_neg_bboxes']
    variances = hyper_params['variances'] 
    pos_threshold = hyper_params["pos_threshold"]
    neg_threshold = hyper_params["neg_threshold"]

    iou_map = bbox_utils.generate_iou(anchors, gt_boxes)
    #
    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    merged_iou_map = tf.reduce_max(iou_map, axis=2) 

    pos_mask = tf.greater(merged_iou_map, pos_threshold)
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
    
    bbox_deltas = bbox_utils.bbox_to_delta(anchors, expanded_gt_boxes) / variances

    bbox_deltas = tf.reshape(bbox_deltas, (batch_size, feature_map_shape, feature_map_shape, anchor_count* 4))
    bbox_labels = tf.reshape(bbox_labels, (batch_size, feature_map_shape, feature_map_shape, anchor_count))
    
    return bbox_deltas, bbox_labels
    
#%%
def dtn_target(roi_bboxes, gt_boxes, gt_labels, hyper_params):
        total_labels = hyper_params["total_labels"]
        total_pos_bboxes = hyper_params["total_pos_bboxes"]
        total_neg_bboxes = hyper_params["total_neg_bboxes"]
        variances = hyper_params["variances"]
        #
        iou_map = bbox_utils.generate_iou(roi_bboxes, gt_boxes)
        #
        max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
        merged_iou_map = tf.reduce_max(iou_map, axis=2)
        pos_mask = tf.greater(merged_iou_map, 0.5)
        pos_mask = randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
        #
        neg_mask = tf.logical_and(tf.less(merged_iou_map, 0.5), tf.greater(merged_iou_map, 0.1))
        neg_mask = randomly_select_xyz_mask(neg_mask, tf.constant([total_neg_bboxes], dtype=tf.int32))
        #
        gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
        expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, axis=-1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
        #
        gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)

        pos_gt_labels = tf.where(pos_mask, gt_labels_map, tf.constant(-1, dtype=tf.int32))
        neg_gt_labels = tf.cast(neg_mask, dtype=tf.int32)

        expanded_gt_labels = pos_gt_labels + neg_gt_labels 
        #
        roi_deltas = bbox_utils.bbox_to_delta(roi_bboxes, expanded_gt_boxes) / variances
        #
        roi_labels = tf.one_hot(expanded_gt_labels, total_labels)
        scatter_indices = tf.tile(tf.expand_dims(roi_labels, -1), (1, 1, 1, 4))
        roi_deltas = scatter_indices * tf.expand_dims(roi_deltas, -2)

        return roi_deltas, roi_labels

#%%
def randomly_select_xyz_mask(mask, select_xyz):
    maxval = tf.reduce_max(select_xyz) * 10
    random_mask = tf.random.uniform(tf.shape(mask), minval=1, maxval=maxval, dtype=tf.int32)
    multiplied_mask = tf.cast(mask, tf.int32) * random_mask
    sorted_mask = tf.argsort(multiplied_mask, direction="DESCENDING")
    sorted_mask_indices = tf.argsort(sorted_mask)
    selected_mask = tf.less(sorted_mask_indices, tf.expand_dims(select_xyz, 1))
    return tf.logical_and(mask, selected_mask)
    