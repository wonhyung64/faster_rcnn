#%%
import tensorflow as tf
import bbox_utils
#%%
def RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params, nms_iou_threshold=0.7, test=False):
        pre_nms_topn = hyper_params["pre_nms_topn"]
        post_nms_topn = hyper_params["train_nms_topn"]
        if test == True: post_nms_topn = hyper_params["test_nms_topn"]
        # nms_iou_threshold = hyper_params["nms_iou_threshold"] 
        variances = hyper_params["variances"]
        total_anchors = hyper_params["feature_map_shape"]**2 * hyper_params["anchor_count"]
        batch_size = tf.shape(rpn_reg_output)[0]

        rpn_reg_output = tf.reshape(rpn_reg_output, (batch_size, total_anchors, 4))
        rpn_cls_output = tf.reshape(rpn_cls_output, (batch_size, total_anchors))
        #
        rpn_reg_output *= variances
        #
        rpn_bboxes = bbox_utils.delta_to_bbox(anchors, rpn_reg_output)

        _, pre_indices = tf.nn.top_k(rpn_cls_output, pre_nms_topn)
        #
        pre_roi_bboxes = tf.gather(rpn_bboxes, pre_indices, batch_dims=1)
        pre_roi_probs = tf.gather(rpn_cls_output, pre_indices, batch_dims=1)
        #
        pre_roi_bboxes = tf.reshape(pre_roi_bboxes, (batch_size, pre_nms_topn, 1, 4))
        pre_roi_probs = tf.reshape(pre_roi_probs, (batch_size, pre_nms_topn, 1))
        #
        
        roi_bboxes, roi_scores, _, _ = tf.image.combined_non_max_suppression(pre_roi_bboxes, pre_roi_probs,
                                                            max_output_size_per_class=post_nms_topn,
                                                            max_total_size = post_nms_topn,
                                                            iou_threshold=nms_iou_threshold)
        #
        return roi_bboxes, roi_scores
# %%
def RoIAlign(roi_bboxes, feature_map, hyper_params):
    pooling_size = hyper_params["pooling_size"]
    batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
    #
    row_size = batch_size * total_bboxes
    #
    pooling_bbox_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), (1, total_bboxes))
    pooling_bbox_indices = tf.reshape(pooling_bbox_indices, (-1, ))
    pooling_bboxes = tf.reshape(roi_bboxes, (row_size, 4))
    #
    pooled_roi = tf.image.crop_and_resize(feature_map,
                                                    pooling_bboxes,
                                                    pooling_bbox_indices,
                                                    pooling_size) 

    pooled_roi = tf.reshape(pooled_roi, (batch_size, total_bboxes,
                                        pooled_roi.shape[1], pooled_roi.shape[2], pooled_roi.shape[3]))
    #
    return pooled_roi

#%%
def Decode(dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params, max_total_size=200, score_threshold=0.7, iou_threshold=0.5):
    batch_size = tf.shape(dtn_reg_output)[0]
    variances = hyper_params["variances"]
    total_labels = hyper_params["total_labels"]

    dtn_reg_output = tf.reshape(dtn_reg_output, (batch_size, -1, total_labels, 4))
    dtn_reg_output *= variances

    expanded_roi_bboxes = tf.tile(tf.expand_dims(roi_bboxes, -2), (1, 1, total_labels, 1))
    
    pred_bboxes = bbox_utils.delta_to_bbox(expanded_roi_bboxes, dtn_reg_output)

    pred_labels_map = tf.expand_dims(tf.argmax(dtn_cls_output, -1), -1)
    pred_labels = tf.where(tf.not_equal(pred_labels_map, 0), dtn_cls_output, tf.zeros_like(dtn_cls_output))
    
    final_bboxes, final_scores, final_labels, _ = tf.image.combined_non_max_suppression(
                        pred_bboxes, pred_labels,
                        max_output_size_per_class = max_total_size,
                        max_total_size = max_total_size,
                        iou_threshold=iou_threshold,
                        score_threshold=score_threshold
                    )
    return final_bboxes, final_labels, final_scores
