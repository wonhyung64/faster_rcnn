#%% 
import os
import time
import tensorflow as tf
from tqdm import tqdm

import utils, model_utils, preprocessing_utils, postprocessing_utils, tune_utils, data_utils, anchor_utils, draw_utils

#%% 
hyper_params = tune_utils.get_hyper_params()
hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])

hyper_params["batch_size"] = batch_size = 1
img_size = (hyper_params["img_size"], hyper_params["img_size"])

dataset, labels = data_utils.fetch_dataset("voc07", "train", img_size)

dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))
data_shapes = ([None, None, None], [None, None], [None])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
dataset = dataset.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
dataset = iter(dataset)

labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)

anchors = anchor_utils.generate_anchors(hyper_params)

#%%
weights_dir = os.getcwd() + r"\atmp"
weights_dir = weights_dir + "\\" + os.listdir(weights_dir)[-1]
weights_dir = r"C:\won\frcnn\atmp1"

rpn_model = model_utils.RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)
rpn_model.load_weights(weights_dir + r'\rpn_weights\weights')

dtn_model = model_utils.DTN(hyper_params)
input_shape = (None, hyper_params['train_nms_topn'], 7, 7, 512)
dtn_model.build(input_shape)
dtn_model.load_weights(weights_dir + r'\frcnn_weights\weights')

#%%
save_dir = os.getcwd()
save_dir = utils.generate_save_dir(save_dir, hyper_params)

progress_bar = tqdm(range(hyper_params['attempts']))

for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    start_time = time.time()
    rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
    roi_bboxes, _ = postprocessing_utils.RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
    pooled_roi = postprocessing_utils.RoIAlign(roi_bboxes, feature_map, hyper_params)
    dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params)
    print("Time taken: %.2fms" % (float(time.time() - start_time)*1000))
    draw_utils.draw_frcnn_output(img, final_bboxes, labels, final_labels, final_scores)
#%%
final_bboxes
final_labels
final_scores

#%%
import bbox_utils
iou = bbox_utils.generate_iou(final_bboxes, gt_boxes)

final_labels

labels
total_labels = hyper_params["total_labels"]
for c in range(total_labels):

c = 15
final_bbox = tf.expand_dims(final_bboxes[final_labels == c], axis=0)
final_label = tf.expand_dims(final_labels[final_labels == c], axis=0)
final_score = tf.expand_dims(final_scores[final_labels == c], axis=0)

gt_box = tf.expand_dims(gt_boxes[gt_labels == c], axis=0)
gt_label = tf.expand_dims(gt_labels[gt_labels == c], axis=0)

iou = bbox_utils.generate_iou(final_bbox, gt_box)

best_iou = tf.reduce_max(iou, axis=1)

pos_num = gt_box.shape[1]

final_bbox
iou = bbox_utils.generate_iou(final_bbox, gt_boxes)