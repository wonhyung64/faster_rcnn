#%% 
import os
import tensorflow as tf
from tqdm import tqdm

import utils, model_utils, preprocessing_utils, postprocessing_utils, tune_utils, data_utils, anchor_utils, draw_utils

#%% 
hyper_params = tune_utils.get_hyper_params()
hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])

hyper_params["batch_size"] = batch_size = 1
img_size = (hyper_params["img_size"], hyper_params["img_size"])

train, labels = data_utils.fetch_dataset("voc07", "train", img_size)

train = train.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))
data_shapes = ([None, None, None], [None, None], [None])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
train = train.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
dataset = iter(train)

labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)

anchors = anchor_utils.generate_anchors(hyper_params)

#%%
weights_dir = os.getcwd() + r"\atmp"
weights_dir = weights_dir + "\\" + os.listdir(weights_dir)[-1]

rpn_model = model_utils.RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)
rpn_model.load_weights(weights_dir + r'\rpn_weights\weights')

dtn_model = model_utils.DTN(hyper_params)
input_shape = (None, hyper_params['train_nms_topn'], 7, 7, 512)
dtn_model.build(input_shape)
dtn_model.load_weights(weights_dir + r'\dtn_weights\weights')

#%%
save_dir = os.getcwd()
save_dir = utils.generate_save_dir(save_dir, hyper_params)

progress_bar = tqdm(range(hyper_params['attempts']))

for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
    roi_bboxes, _ = postprocessing_utils.RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
    pooled_roi = postprocessing_utils.RoIAlign(roi_bboxes, feature_map, hyper_params)
    dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params)

    draw_utils.draw_frcnn_output(img, final_bboxes, labels, final_labels, final_scores)
#%%

