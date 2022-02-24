#%% 
import os
import time
import tensorflow as tf
import numpy as np

from PIL import Image
from tqdm import tqdm

import utils, model_utils, preprocessing_utils, postprocessing_utils, anchor_utils, test_utils

from PIL import ImageDraw
import matplotlib.pyplot as plt
#%% 
hyper_params = utils.get_hyper_params()
hyper_params['anchor_count'] = len(hyper_params['anchor_ratios']) * len(hyper_params['anchor_scales'])

hyper_params["batch_size"] = batch_size = 1
img_size = (hyper_params["img_size"], hyper_params["img_size"])
dataset_name = hyper_params["dataset_name"]

if dataset_name == "ship":
    import ship
    dataset, labels = ship.fetch_dataset(dataset_name, "test", img_size)
    dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing_ship(x, y, z, w))
else:
    import data_utils
    dataset, labels = data_utils.fetch_dataset(dataset_name, "test", img_size)
    dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))

data_shapes = ([None, None, None], [None, None], [None])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
dataset = dataset.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
dataset = iter(dataset)

labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)

anchors = anchor_utils.generate_anchors(hyper_params)

#%%
weights_dir = os.getcwd() + "/frcnn_atmp"
weights_dir = weights_dir + "/" + os.listdir(weights_dir)[0]

rpn_model = model_utils.RPN(hyper_params)
input_shape = (None, 500, 500, 3)
rpn_model.build(input_shape)
rpn_model.load_weights(weights_dir + '/rpn_weights/weights')

dtn_model = model_utils.DTN(hyper_params)
input_shape = (None, hyper_params['train_nms_topn'], 7, 7, 512)
dtn_model.build(input_shape)
dtn_model.load_weights(weights_dir + '/dtn_weights/weights')

#%%
# save_dir = os.getcwd()
# save_dir = utils.generate_save_dir(save_dir, hyper_params)

total_time = []
mAP = []
optimal_threshold = []
threshold_lst = np.arange(0.5, 1.0, 0.05)
num = 0
progress_bar = tqdm(range(278))
for _ in progress_bar:
    img, gt_boxes, gt_labels = next(dataset)
    start_time = time.time()
    rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)

    roi_bboxes, _ = postprocessing_utils.RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
    pooled_roi = postprocessing_utils.RoIAlign(roi_bboxes, feature_map, hyper_params)
    dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)

    best_threshold = 0.
    best_AP = 0.
    for threshold in threshold_lst:
        final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params, iou_threshold=threshold)
        AP = test_utils.calculate_AP(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params)
        # AP = test_utils.calculate_AP50(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params)
        if AP >= best_AP: 
            best_threshold = threshold
            best_AP = AP

    mAP.append(best_AP)
    optimal_threshold.append(best_threshold)

    # time_ = float(time.time() - start_time)*1000
    # AP = test_utils.calculate_AP50(final_bboxes, final_labels, gt_boxes, gt_labels, hyper_params)
    # test_utils.draw_dtn_output(img, final_bboxes, labels, final_labels, final_scores, )
    # print(num)
    # total_time.append(time_)
    # mAP.append(AP)
    # num += 1

    
print("mAP: %.2f" % (tf.reduce_mean(mAP)))
# print("Time taken: %.2fms" % (tf.reduce_mean(total_time)))

#%%
def draw_custom_img(img_dir):
    image = Image.open(img_dir)
    image_ = np.array(image)
    image_ = tf.convert_to_tensor(image_)
    image_ = tf.image.resize(image_, (500,500))/ 255
    img = tf.expand_dims(image_, axis=0)
    rpn_reg_output, rpn_cls_output, feature_map = rpn_model(img)
    roi_bboxes, _ = postprocessing_utils.RoIBBox(rpn_reg_output, rpn_cls_output, anchors, hyper_params)
    pooled_roi = postprocessing_utils.RoIAlign(roi_bboxes, feature_map, hyper_params)
    dtn_reg_output, dtn_cls_output = dtn_model(pooled_roi)
    final_bboxes, final_labels, final_scores = postprocessing_utils.Decode(dtn_reg_output, dtn_cls_output, roi_bboxes, hyper_params)
    test_utils.draw_frcnn_output(img, final_bboxes, labels, final_labels, final_scores)

# test_utils.draw_custom_img("C:/won/test9.jpg")
#%%