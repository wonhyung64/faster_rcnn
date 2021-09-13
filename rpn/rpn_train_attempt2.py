#%%
# 수정사항
#  - preprocessing 함수 모듈화
#  - generating anchors 함수 모듈화
#%% Module

import os
import math

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

from utils import data_utils, bbox_utils

#%% OPTION 
batch_size = 8
epochs = 50
load_weights = False

hyper_params = {"img_size": 500,
                "feature_map_shape": 31,
                "anchor_ratios": [1., 2., 1./2.],
                "anchor_scales": [128, 256, 512],
                "pre_nms_topn": 6000,
                "train_nms_topn": 1500,
                "test_nms_topn": 300,
                "nms_iou_threshold": 0.7,
                "total_pos_bboxes": 128,
                "total_neg_bboxes": 128,
                "pooling_size": (7,7),
                "variances": [0.1, 0.1, 0.2, 0.2],
                }

hyper_params["anchor_count"] = len(hyper_params["anchor_ratios"]) * len(hyper_params["anchor_scales"])


#%% DATA IMPORT
train_data, dataset_info = tfds.load("voc/2007", split="train+validation", data_dir = "C:\won\data\pascal_voc\\tensorflow_datasets", with_info=True)
val_data, _ = tfds.load("voc/2007", split="test", data_dir = "C:\won\data\pascal_voc\\tensorflow_datasets", with_info=True)
# under bar 로 데이터를 불러오지 않기

train_total_items = dataset_info.splits["train"].num_examples + dataset_info.splits["validation"].num_examples
val_total_items = dataset_info.splits["test"].num_examples


#%% EDA
labels = dataset_info.features["labels"].names
hyper_params["total_labels"] = len(labels) + 1


#%% DATA PREPROCESSING
img_size = hyper_params["img_size"]

train_data = train_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, apply_augmentation=True))
val_data = val_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))


# %%
data_shapes = ([None, None, None], [None, None], [None,])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
train_data = train_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
# batch size = 8 한번에 8개의 사진을 사용

val_data = val_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)


#%% ANCHOR
anchors = bbox_utils.generate_anchors(hyper_params)


#%% Generating Region Proposal DEF
def rpn_generator(dataset, anchors, hyper_params):
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            bbox_deltas, bbox_labels = calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params)
            yield img, (bbox_deltas, bbox_labels)


def calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params):
    batch_size = tf.shape(gt_boxes)[0] # gt_boxes 는 [batch_size, 이미지 데이터 한장에 있는 라벨의 갯수(4개) , 좌표(4)]
    feature_map_shape = hyper_params['feature_map_shape'] # feature_map_shape = 31
    anchor_count = hyper_params['anchor_count'] # anchor_count = 3 * 3 
    total_pos_bboxes = hyper_params['total_pos_bboxes'] # 이게 뭘 나타내는건지 알 수 없음.. 128
    total_neg_bboxes = hyper_params['total_neg_bboxes'] # 이것도 마찬가지 .. 128
    variances = hyper_params['variances'] # variances 가 무슨값인지 알 수 없음

    # Generating IoU map
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(anchors, 4, axis=-1) # C X C  X anchor_count 개의 reference anchors 의 x, y 좌표
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1) # gt_boxes에 있는 박스들 각각의 x, y 좌표
    
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1) # tf.squeeze : 텐서에서 사이즈 차원이 1이 아닌 부분만 짜낸다.
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis = -1)
    
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1])) # tf.transpose : 텐서를 [] 순서의 모양으로 transpose
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))
    
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area)
    
    iou_map = intersection_area / union_area 
    # 8장의 사진에 대한, C X C X 9 개의 reference anchors 와, ground truth box n개(최대4) 와의 IoU계산
    #
    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # 각 사진의 reference anchor 에서 IoU가 가장 높게 나오는 gt_box의 index 추출 (8, 8649, 1)

    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    # 각 사진의 gt_box에서 IoU가 가장 높게 나오는 reference anchor 의 index 추출 (8, 1, 4)
    #
    merged_iou_map = tf.reduce_max(iou_map, axis=2) 
    # 8장의 사진 에서 하나의 reference anchor와 4개의 gt_boxes 들 중 가장 높은값만 남기기
    #
    pos_mask = tf.greater(merged_iou_map, 0.7)
    # 각 사진에서 가장 높은 IoU가 threshold 보다 높은지 
    #
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    # 왜 -1? : gt_labels 중 라벨이 없는 값은 -1 로 입력
    
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    # valid_indices 에서 label이 있는 부분의 tensor_index 반환
    
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    # 8장의 사진에 15개의 라벨이 있고 이들과의 IoU가 가장높은 사진에서의 reference anchor index 반환
    #

    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)
    # 8장의 사진 에서 라벨이 존재하는 사진의 index와 해당 라벨의 gt_box와 가장 높은 IoU를 가지는 reference acnhor의 index 반환 
    max_pos_mask = tf.scatter_nd(indices=scatter_bbox_indices, updates=tf.fill((tf.shape(valid_indices)[0], ), True), shape=tf.shape(pos_mask))
    # 8장의 사진 각각의 reference anchors에서 gt_box와의 가장 높은 IoU 를 가지는 reference anchor의 index 만 True
    pos_mask = tf.logical_or(pos_mask, max_pos_mask)
    # pos_mask에 threshold 이상의 IoU를 가지는 reference anchor 와, 가장 높은 IoU를 가지는 reference anchor 만 True 반환
    pos_mask = randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
    #
    pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1)
    # 8장의 pos_mask 에 대해서 각각 true의 개수 반환
    neg_count = (total_pos_bboxes + total_neg_bboxes) - pos_count
    
    neg_mask = tf.logical_and(tf.less(merged_iou_map, 0.3), tf.logical_not(pos_mask))
    # 8장의 사진에서 Iou가 0.3보다 작고 pos_mask가 false인 부분만 False
    neg_mask = randomly_select_xyz_mask(neg_mask, neg_count)
    #
    
    pos_labels = tf.where(pos_mask, tf.ones_like(pos_mask, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32))
    # ?
    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)
    # 정리 자료에서 cls loss의 p*_i
    gt_boxes_map = tf.gather(params=gt_boxes, indices=max_indices_each_row, batch_dims=1)
    # 8장의 사진에서 8649개의 reference anchor 중 4개의 gt_box와 IoU가 가장 높은 IoU값 4개의 gt_box에 대해 각각 반환

    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    # 정리 자료에서 reg loss의 p*_i
    
    bbox_width = anchors[..., 3] - anchors[..., 1]
    bbox_height = anchors[..., 2] - anchors[...,0]
    bbox_ctr_x = anchors[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = anchors[..., 0] + 0.5 * bbox_height
    
    gt_width = expanded_gt_boxes[..., 3] - expanded_gt_boxes[..., 1]
    gt_height = expanded_gt_boxes[..., 2] - expanded_gt_boxes[..., 0]
    gt_ctr_x = expanded_gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = expanded_gt_boxes[..., 0] + 0.5 * gt_height
    
    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / bbox_width))
    delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.math.log(gt_height / bbox_height))
    
    bbox_deltas = tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1) / variances
    bbox_deltas = tf.reshape(bbox_deltas, (batch_size, feature_map_shape, feature_map_shape, anchor_count * 4))
    bbox_labels = tf.reshape(bbox_labels, (batch_size, feature_map_shape, feature_map_shape, anchor_count))
    
    return bbox_deltas, bbox_labels


def randomly_select_xyz_mask(mask, select_xyz):
    maxval = tf.reduce_max(select_xyz) * 10
    random_mask = tf.random.uniform(tf.shape(mask), minval=1, maxval=maxval, dtype=tf.int32)
    multiplied_mask = tf.cast(mask, tf.int32) * random_mask
    sorted_mask = tf.argsort(multiplied_mask, direction="DESCENDING")
    sorted_mask_indices = tf.argsort(sorted_mask)
    selected_mask = tf.less(sorted_mask_indices, tf.expand_dims(select_xyz, 1))
    return tf.logical_and(mask, selected_mask)
    

#%% Generating Region Proposal
rpn_train_feed = rpn_generator(train_data, anchors, hyper_params)
rpn_val_feed = rpn_generator(val_data, anchors, hyper_params)


#%% RPN MODEL with VGG-16
img_size = hyper_params['img_size']

base_model = VGG16(include_top=False, input_shape=(img_size, img_size, 3))
# include_top ?
# input shape : (img_size) x  (img_size) x 3channel

feature_extractor = base_model.get_layer("block5_conv3")
feature_extractor.output.shape
# VGG16의 마지막 합성곱 층인 "blcok5_conv3" 를 feature_extractor에 할당

output = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="rpn_conv")(feature_extractor.output)
output.shape
# output의 차원이 512 = feature map 하나에 512 차원의 정보를 가짐.

rpn_cls_output = Conv2D(filters=hyper_params["anchor_count"], kernel_size=(1,1), activation="sigmoid", name="rpn_cls")(output)
rpn_cls_output.shape
# rpn model 에서 나오는 classification layer

rpn_reg_output = Conv2D(filters=hyper_params["anchor_count"]*4, kernel_size=(1,1), activation="linear", name="rpn_reg")(output)
rpn_reg_output.shape
# rpn model 에서 나오는 regression layer

rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])
# rpn model 구축

# rpn_model.summary()
# feature_extractor 를 왜 return ?


#%% Regression Loss Function
def rpn_reg_loss(*args):
    y_pred, y_true = args if len(args) == 2 else args[0]
    #
    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
    # Huber : SmoothL1 loss function

    loss_for_all = loss_fn(y_true[0], y_pred[0])
    loss_for_all = tf.reduce_sum(loss_for_all, axis=-1)
    # sum of SmoothL1
    
    pos_cond = tf.reduce_any(tf.not_equal(y_true[1], tf.constant(0.0)), axis=-1)
    # tf.reduce_any?
    
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    # positive label
    
    #
    loc_loss = tf.reduce_sum(pos_mask * loss_for_all)

    total_pos_bboxes = tf.maximum(1.0, tf.reduce_sum(pos_mask))

    return loc_loss / total_pos_bboxes


#%% Objectness Loss Function
def rpn_cls_loss(*args):
    y_pred, y_true = args if len(args) == 2 else args[0]

    indices = tf.where(tf.not_equal(y_true[1], tf.constant(-1.0, dtype=tf.float32)))

    target = tf.gather_nd(y_true[1], indices)
    output = tf.gather_nd(y_pred[1], indices)
    # tf.gather_nd ?
    
    lf = tf.losses.BinaryCrossentropy()
    return lf(target, output)


# %% Training
rpn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-5),
                  loss=[rpn_reg_loss, rpn_cls_loss])

main_path = "C:/Users/USER/Documents/GitHub/faster_rcnn/rpn"
if not os.path.exists(main_path):
    os.makedirs(main_path)
rpn_model_path = os.path.join(main_path, "{}_{}_model_weights_attempt2.h5".format("rpn", "vgg16"))

checkpoint_callback = ModelCheckpoint(rpn_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)

step_size_train = math.ceil(train_total_items / batch_size)
step_size_val = math.ceil(val_total_items / batch_size)


#%%
rpn_model.fit(rpn_train_feed,
              steps_per_epoch=step_size_train,
              validation_data=rpn_val_feed,
              validation_steps=step_size_val,
              epochs=epochs,
              callbacks=[checkpoint_callback])



